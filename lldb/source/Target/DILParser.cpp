//===-- DILParser.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/DILParser.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Token.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Target/DILAst.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Host.h"

namespace {

const char* kInvalidOperandsToUnaryExpression =
    "invalid argument type {0} to unary expression";

const char* kInvalidOperandsToBinaryExpression =
    "invalid operands to binary expression ({0} and {1})";

const char* kValueIsNotConvertibleToBool =
    "value of type {0} is not contextually convertible to 'bool'";

template <typename T>
constexpr unsigned type_width() {
  return static_cast<unsigned>(sizeof(T)) * CHAR_BIT;
}

inline void TokenKindsJoinImpl(std::ostringstream& os,
                               clang::tok::TokenKind k) {
  os << "'" << clang::tok::getTokenName(k) << "'";
}

template <typename... Ts>
inline void TokenKindsJoinImpl(std::ostringstream& os, clang::tok::TokenKind k,
                               Ts... ks) {
  TokenKindsJoinImpl(os, k);
  os << ", ";
  TokenKindsJoinImpl(os, ks...);
}

template <typename... Ts>
inline std::string TokenKindsJoin(clang::tok::TokenKind k, Ts... ks) {
  std::ostringstream os;
  TokenKindsJoinImpl(os, k, ks...);

  return os.str();
}

}  // namespace

namespace lldb_private {

std::string TypeDescription(CompilerType type) {
  auto name = type.GetTypeName();
  auto canonical_name =
      type.GetCanonicalType().GetTypeName();
  if (name.IsEmpty() || canonical_name.IsEmpty()) {
    return "''";  // should not happen
  }
  if (name == canonical_name) {
    return llvm::formatv("'{0}'", name);
  }
  return llvm::formatv("'{0}' (aka '{1}')", name, canonical_name);
}

static CompilerType GetBasicType(std::shared_ptr<ExecutionContextScope> ctx,
                                 lldb::BasicType basic_type) {
  static std::unordered_map<lldb::BasicType, CompilerType> basic_types;
  auto type = basic_types.find(basic_type);
  if (type != basic_types.end()) {
    std::string type_name((type->second).GetTypeName().AsCString());
    // Only return the found type if it's valid.
    if (type_name != "<invalid>")
      return type->second;
  }

  lldb::TargetSP target_sp = ctx->CalculateTarget();
  if (target_sp) {
    for (auto type_system_sp : target_sp->GetScratchTypeSystems())
      if (auto compiler_type = type_system_sp->GetBasicTypeFromAST(basic_type)){
        basic_types.insert({basic_type, compiler_type});
        return compiler_type;
      }
  }
  CompilerType empty_type;
  return empty_type;
}

static lldb::BasicType GetPtrDiffType(std::shared_ptr<ExecutionContextScope> ctx)
{
  lldb::TargetSP target_sp = ctx->CalculateTarget();
  llvm::Triple triple(
      llvm::Twine(target_sp->GetArchitecture().GetTriple().str()));

  if (triple.isOSWindows()) {
    return triple.isArch64Bit() ? lldb::eBasicTypeLongLong
                                : lldb::eBasicTypeInt;
  } else {
    return triple.isArch64Bit() ? lldb::eBasicTypeLong : lldb::eBasicTypeInt;
  }
}

static std::unique_ptr<BuiltinFunctionDef> GetBuiltinFunctionDef(
    std::shared_ptr<ExecutionContextScope> ctx, const std::string& identifier) {
  //
  // __log2(unsigned int x) -> unsigned int
  //
  //   Calculates the log2(x).
  //
  if (identifier == "__log2") {
    CompilerType return_type = GetBasicType(ctx, lldb::eBasicTypeUnsignedInt);
    std::vector<CompilerType> arguments = {
      GetBasicType(ctx, lldb::eBasicTypeUnsignedInt),
    };
    return std::make_unique<BuiltinFunctionDef>(identifier, return_type,
                                                std::move(arguments));
  }
  //
  // __findnonnull(T* ptr, long long buffer_size) -> int
  //
  //   Finds the first non-null object pointed by `ptr`. `ptr` is treated as an
  //   array of pointers of size `buffer_size`.
  //
  if (identifier == "__findnonnull") {
    auto return_type = GetBasicType(ctx, lldb::eBasicTypeInt);
    std::vector<CompilerType> arguments = {
        // The first argument should actually be "T*", but we don't support
        // templates here.
        // HACK: Void means "any" and we'll check in runtime. The argument will
        // be passed as is without any conversions.
      GetBasicType(ctx, lldb::eBasicTypeVoid),
      GetBasicType(ctx,lldb::eBasicTypeLongLong),
    };
    return std::make_unique<BuiltinFunctionDef>(identifier, return_type,
                                                std::move(arguments));
  }
  // Not a builtin function.
  return nullptr;
}

bool CompareTypes(CompilerType lhs, CompilerType rhs) {
  if (lhs == rhs)
    return true;

  const ConstString lhs_name = lhs.GetFullyUnqualifiedType().GetTypeName();
  const ConstString rhs_name = rhs.GetFullyUnqualifiedType().GetTypeName();
  return lhs_name == rhs_name;
}

static const char* GetTypeTag(CompilerType type) {
  switch (type.GetTypeClass()) {
      // clang-format off
    case lldb::eTypeClassClass:       return "class";
    case lldb::eTypeClassEnumeration: return "enum";
    case lldb::eTypeClassStruct:      return "struct";
    case lldb::eTypeClassUnion:       return "union";
      // clang-format on
    default:
      return "unknown";
  }
}

static bool GetPathToBaseType(CompilerType type, CompilerType target_base,
                              std::vector<uint32_t>* path,
                              uint64_t* offset) {
  if (CompareTypes(type, target_base)) {
    return true;
  }

  uint32_t bit_offset = 0;
  uint32_t num_non_empty_bases = 0;
  uint32_t num_direct_bases =
      type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    auto member_base_type =
        type.GetDirectBaseClassAtIndex(i, &bit_offset);
    if (GetPathToBaseType(member_base_type, target_base, path, offset)) {
      if (path) {
        path->push_back(num_non_empty_bases);
      }
      if (offset) {
        *offset += bit_offset / 8u;
      }
      return true;
    }
    if (member_base_type.GetNumFields() > 0) {
      num_non_empty_bases++;
    }
  }

  return false;
}

struct MemberInfo {
  std::optional<std::string> name;
  CompilerType type;
  bool is_bitfield;
  uint32_t bitfield_size_in_bits;
  bool is_synthetic;

  explicit operator bool() const { return type.IsValid(); }
};


static uint32_t GetNumberOfNonEmptyBaseClasses(CompilerType type) {
  // Go through the base classes and count non-empty ones.
  uint32_t ret = 0;
  uint32_t num_direct_bases = type.GetNumDirectBaseClasses();

  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    uint32_t bit_offset;
    CompilerType base_type = type.GetDirectBaseClassAtIndex(i, &bit_offset);
    if (base_type.GetNumFields() > 0 ||
        GetNumberOfNonEmptyBaseClasses(base_type) > 0) {
      ret += 1;
    }
  }
  return ret;
}

static MemberInfo GetFieldWithNameIndexPath(lldb::ValueObjectSP lhs_val_sp,
                                            CompilerType type,
                                            const std::string& name,
                                            std::vector<uint32_t>* idx,
                                            CompilerType empty_type,
                                            bool use_synthetic) {
  bool is_synthetic = false;
  // Go through the fields first.
  uint32_t num_fields = type.GetNumFields();
  for (uint32_t i = 0; i < num_fields; ++i) {
    uint64_t bit_offset = 0;
    uint32_t bitfield_bit_size = 0;
    bool is_bitfield = false;
    std::string name_sstr;
    CompilerType field_type (
        type.GetFieldAtIndex(
            i, name_sstr, &bit_offset, &bitfield_bit_size, &is_bitfield));
    auto field_name = name_sstr.length() == 0 ? std::optional<std::string>()
                                              : name_sstr;
    if (field_type.IsValid()) {
      struct MemberInfo field = {field_name, field_type, is_bitfield,
        bitfield_bit_size, is_synthetic };

      // Name can be null if this is a padding field.
      if (field.name == name) {
        if (idx) {
          assert(idx->empty());
          // Direct base classes are located before fields, so field members
          // needs to be offset by the number of base classes.
          idx->push_back(i + GetNumberOfNonEmptyBaseClasses(type));
        }
        return field;
      } else if (field.type.IsAnonymousType()) {
        // Every member of an anonymous struct is considered to be a member of
        // the enclosing struct or union. This applies recursively if the
        // enclosing struct or union is also anonymous.
        //
        //  struct S {
        //    struct {
        //      int x;
        //    };
        //  } s;
        //
        //  s.x = 1;

        assert(!field.name && "Field should be unnamed.");

        auto field_in_anon_type =
            GetFieldWithNameIndexPath(lhs_val_sp, field.type, name, idx,
                                      empty_type, use_synthetic);
        if (field_in_anon_type) {
          if (idx) {
            idx->push_back(i + GetNumberOfNonEmptyBaseClasses(type));
          }
          return field_in_anon_type;
        }
      }
    }
  }

  // LLDB can't access inherited fields of anonymous struct members.
  if (type.IsAnonymousType()) {
    return {{}, empty_type, false, 0};
  }

  // Go through the base classes and look for the field there.
  uint32_t num_non_empty_bases = 0;
  uint32_t num_direct_bases = type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    uint32_t bit_offset;
    auto base = type.GetDirectBaseClassAtIndex(i, &bit_offset);
    auto field = GetFieldWithNameIndexPath(lhs_val_sp, base, name, idx,
                                           empty_type, use_synthetic);
    if (field) {
      if (idx) {
        idx->push_back(num_non_empty_bases);
      }
      return field;
    }
    if (base.GetNumFields() > 0) {
      num_non_empty_bases += 1;
    }
  }

  // Check for synthetic member
  if (lhs_val_sp && use_synthetic) {
    lldb::ValueObjectSP child_valobj_sp = lhs_val_sp->GetSyntheticValue();
    if (child_valobj_sp) {
      is_synthetic = true;
      uint32_t child_idx = child_valobj_sp->GetIndexOfChildWithName(name);
      child_valobj_sp = child_valobj_sp->GetChildMemberWithName(name);
      CompilerType field_type = child_valobj_sp->GetCompilerType();
      if (field_type.IsValid()) {
        struct MemberInfo field = {name, field_type, false, 0, is_synthetic};
        if (idx) {
          assert(idx->empty());
          idx->push_back(child_idx);
        }
        return field;
      }
    }
  }

  return {{}, empty_type, false, 0};
}

static std::tuple<MemberInfo, std::vector<uint32_t>> GetMemberInfo (
    lldb::ValueObjectSP lhs_val_sp, CompilerType type, const std::string& name,
    bool use_synthetic) {
  std::vector<uint32_t> idx;
  CompilerType empty_type;
  MemberInfo member =
      GetFieldWithNameIndexPath(lhs_val_sp, type, name, &idx, empty_type,
                                use_synthetic);
  std::reverse(idx.begin(), idx.end());
  return {member, std::move(idx)};
}


DILSourceManager::DILSourceManager(std::string expr) : m_expr(std::move(expr))
{
  // This holds a DILSourceManager and all of its dependencies.
  m_smff = std::make_unique<clang::SourceManagerForFile>("<expr>", m_expr);

  // Disable default diagnostics reporting.
  // TODO(werat): Add custom consumer to keep track of errors.
  clang::DiagnosticsEngine& de = m_smff->get().getDiagnostics();
  de.setClient(new clang::IgnoringDiagConsumer);
}

std::shared_ptr<DILSourceManager> DILSourceManager::Create(std::string expr) {
  return std::shared_ptr<DILSourceManager>(new DILSourceManager(
      std::move(expr)));
};

static const char* ToString(TypeDeclaration::TypeSpecifier type_spec) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  switch (type_spec) {
      // clang-format off
    case TypeSpecifier::kVoid:       return "void";
    case TypeSpecifier::kBool:       return "bool";
    case TypeSpecifier::kChar:       return "char";
    case TypeSpecifier::kShort:      return "short";
    case TypeSpecifier::kInt:        return "int";
    case TypeSpecifier::kLong:       return "long";
    case TypeSpecifier::kLongLong:   return "long long";
    case TypeSpecifier::kFloat:      return "float";
    case TypeSpecifier::kDouble:     return "double";
    case TypeSpecifier::kLongDouble: return "long double";
    case TypeSpecifier::kWChar:      return "wchar_t";
    case TypeSpecifier::kChar16:     return "char16_t";
    case TypeSpecifier::kChar32:     return "char32_t";
      // clang-format on
    default:
      assert(false && "invalid type specifier");
      return nullptr;
  }
}

static const char* ToString(TypeDeclaration::SignSpecifier sign_spec) {
  using SignSpecifier = TypeDeclaration::SignSpecifier;
  switch (sign_spec) {
      // clang-format off
    case SignSpecifier::kSigned:   return "signed";
    case SignSpecifier::kUnsigned: return "unsigned";
      // clang-format on
    default:
      assert(false && "invalid sign specifier");
      return nullptr;
  }
}

bool IsSmartPtrType(CompilerType type) {
  // Regular expressions are mirrored from LLDB:
  // https://github.com/llvm/llvm-project/blob/release/13.x/lldb/source/Plugins\
/Language/CPlusPlus/CPlusPlusLanguage.cpp#L614-L634
  static llvm::Regex k_libcxx_std_unique_ptr_regex(
      "^std::__[[:alnum:]]+::unique_ptr<.+>(( )?&)?$");
  static llvm::Regex k_libcxx_std_shared_ptr_regex(
      "^std::__[[:alnum:]]+::shared_ptr<.+>(( )?&)?$");
  static llvm::Regex k_libcxx_std_weak_ptr_regex(
      "^std::__[[:alnum:]]+::weak_ptr<.+>(( )?&)?$");
  //
  static llvm::Regex k_libcxx_std_unique_ptr_regex_2(
      "^std::unique_ptr<.+>(( )?&)?$");
  static llvm::Regex k_libcxx_std_shared_ptr_regex_2(
      "^std::shared_ptr<.+>(( )?&)?$");
  static llvm::Regex k_libcxx_std_weak_ptr_regex_2(
      "^std::weak_ptr<.+>(( )?&)?$");
  //
  llvm::StringRef name = type.GetTypeName();
  return k_libcxx_std_unique_ptr_regex.match(name) ||
         k_libcxx_std_shared_ptr_regex.match(name) ||
         k_libcxx_std_weak_ptr_regex.match(name)   ||
         k_libcxx_std_unique_ptr_regex_2.match(name) ||
         k_libcxx_std_shared_ptr_regex_2.match(name) ||
         k_libcxx_std_weak_ptr_regex_2.match(name);
}

bool IsArrayType(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.IsArrayType(nullptr, nullptr, nullptr);
}

bool IsInteger(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetTypeInfo() & lldb::eTypeIsInteger;
}

bool IsScalar(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetTypeInfo() & lldb::eTypeIsScalar;
}

bool IsFloat(CompilerType type) {
  if (!type.IsValid())
    return false;
  return type.GetTypeInfo() & lldb::eTypeIsFloat;
}

bool IsEnum(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetTypeInfo() & lldb::eTypeIsEnumeration;
}

bool IsScopedEnum(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.IsScopedEnumerationType();
}

bool IsUnscopedEnum(CompilerType type) {
  return IsEnum(type) && !IsScopedEnum(type);
}

static bool IsScalarOrUnscopedEnum(CompilerType type) {
  return IsScalar(type) || IsUnscopedEnum(type);
}

bool IsIntegerOrUnscopedEnum(CompilerType type) {
  return IsInteger(type) || IsUnscopedEnum(type);
}

static bool IsPromotableIntegerType(CompilerType type) {
  // Unscoped enums are always considered as promotable, even if their
  // underlying type does not need to be promoted (e.g. "int").
  if (IsUnscopedEnum(type)) {
    return true;
  }

  switch (type.GetCanonicalType().GetBasicTypeEnumeration()) {
    case lldb::eBasicTypeBool:
    case lldb::eBasicTypeChar:
    case lldb::eBasicTypeSignedChar:
    case lldb::eBasicTypeUnsignedChar:
    case lldb::eBasicTypeShort:
    case lldb::eBasicTypeUnsignedShort:
    case lldb::eBasicTypeWChar:
    case lldb::eBasicTypeSignedWChar:
    case lldb::eBasicTypeUnsignedWChar:
    case lldb::eBasicTypeChar16:
    case lldb::eBasicTypeChar32:
      return true;

    default:
      return false;
  }
}

static bool IsEnumerationIntegerTypeSigned(CompilerType type) {
  if (type.IsValid()) {
    return type.GetEnumerationIntegerType().GetTypeInfo()
        & lldb::eTypeIsSigned;
  }
  return false;
}

bool IsSigned(CompilerType type) {
  if (IsEnum(type)) {
    return IsEnumerationIntegerTypeSigned(type);
  }
  return type.GetTypeInfo() & lldb::eTypeIsSigned;
}

bool IsPointerType(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.IsPointerType();
}

bool IsNullPtrType(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetCanonicalType().GetBasicTypeEnumeration() ==
      lldb::eBasicTypeNullPtr;
}

bool IsReferenceType(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.IsReferenceType();
}

static CompilerType GetPointeeType(CompilerType type) {
  CompilerType bad_type;
  if (!type.IsValid())
    return bad_type;

  CompilerType c_type = type.GetPointeeType();
  return c_type;
}

static bool IsPointerToVoid(CompilerType type) {
  if (!type.IsValid())
    return false;

  return IsPointerType(type) && GetPointeeType(type).GetBasicTypeEnumeration()
      == lldb::eBasicTypeVoid;
}

bool IsBool(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetCanonicalType().GetBasicTypeEnumeration() ==
      lldb::eBasicTypeBool;
}

static bool IsRecordType(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.GetCanonicalType().GetTypeClass() &
      (lldb::eTypeClassClass | lldb::eTypeClassStruct | lldb::eTypeClassUnion);
}


// Checks whether `target_base` is a virtual base of `type` (direct or
// indirect). If it is, stores the first virtual base type on the path from
// `type` to `target_type`.
static bool IsVirtualBase(CompilerType type, CompilerType target_base,
                          CompilerType* virtual_base,
                          bool carry_virtual = false) {
  if (CompareTypes(type, target_base)) {
    return carry_virtual;
  }

  if (!carry_virtual) {
    uint32_t num_virtual_bases = type.GetNumVirtualBaseClasses();
    for (uint32_t i = 0; i < num_virtual_bases; ++i) {
      uint32_t bit_offset;
      auto base = type.GetVirtualBaseClassAtIndex(i, &bit_offset);
      if (IsVirtualBase(base, target_base, virtual_base,
                        /*carry_virtual*/ true)) {
        if (virtual_base) {
          *virtual_base = base;
        }
        return true;
      }
    }
  }

  uint32_t num_direct_bases = type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    uint32_t bit_offset;
    auto base = type.GetDirectBaseClassAtIndex(i, &bit_offset);
    if (IsVirtualBase(base, target_base, virtual_base, carry_virtual)) {
      return true;
    }
  }

  return false;
}

static bool IsPolymorphicClass(CompilerType type) {
  if (!type.IsValid())
    return false;

  return type.IsPolymorphicClass();
}

static bool IsContextuallyConvertibleToBool(CompilerType type) {
  if (!type.IsValid())
    return false;

  return IsScalar(type) || IsUnscopedEnum(type) || IsPointerType(type)
      || IsNullPtrType(type) || IsArrayType(type);
}

static CompilerType GetTemplateArgumentType(uint32_t idx, CompilerType this_type)
{
  CompilerType bad_type;
  if (!this_type.IsValid())
    return bad_type;

  CompilerType type;
  const bool expand_pack = true;
  switch(this_type.GetTemplateArgumentKind(idx, true)) {
    case lldb::eTemplateArgumentKindType:
      type = this_type.GetTypeTemplateArgument(idx, expand_pack);
      break;
    case lldb::eTemplateArgumentKindIntegral:
      type = this_type.GetIntegralTemplateArgument(idx, expand_pack)->type;
      break;
    default:
      break;
  }
  if (type.IsValid())
    return type;
  return bad_type;
}

static CompilerType GetSmartPtrPointeeType(CompilerType type) {
  assert(
      IsSmartPtrType(type) &&
      "the type should be a smart pointer (std::unique_ptr, std::shared_ptr "
      "or std::weak_ptr");

  return GetTemplateArgumentType(0, type);
}


static CompilerType GetPointerType(CompilerType type) {
  CompilerType bad_type;
  if (!type.IsValid())
    return bad_type;

  return type.GetPointerType();
}

static CompilerType GetDereferencedType(CompilerType type) {
  CompilerType bad_type;
  if (!type.IsValid())
    return bad_type;

  return type.GetNonReferenceType();
}


static CompilerType GetEnumerationIntegerType(CompilerType type) {
  CompilerType bad_type;
  if (type.IsValid()) {
    return type.GetEnumerationIntegerType();
  }
  return bad_type;;
}

static CompilerType GetReferenceType(CompilerType type) {
  CompilerType bad_type;
  if (!type.IsValid())
    return bad_type;

  return type.GetLValueReferenceType();
}

static CompilerType GetUnqualifiedType(CompilerType type) {
  CompilerType bad_type;
  if (!type.IsValid())
    return bad_type;

  return type.GetFullyUnqualifiedType();
}

static bool TokenEndsTemplateArgumentList(const clang::Token& token) {
  // Note: in C++11 ">>" can be treated as "> >" and thus be a valid token
  // for the template argument list.
  return token.isOneOf(clang::tok::comma, clang::tok::greater,
                       clang::tok::greatergreater);
}

static ExprResult InsertSmartPtrToPointerConversion(ExprResult expr) {
  auto expr_type = expr->result_type_deref();

  assert(
      IsSmartPtrType(expr_type) &&
      "an argument to smart-ptr-to-pointer conversion must be a smart pointer");

  return std::make_unique<SmartPtrToPtrDecay>(
      expr->location(), GetPointerType(GetSmartPtrPointeeType(expr_type)),
      std::move(expr));
}

static ExprResult InsertArrayToPointerConversion(ExprResult expr) {
  assert(IsArrayType(expr->result_type_deref()) &&
         "an argument to array-to-pointer conversion must be an array");

  // TODO(werat): Make this an explicit array-to-pointer conversion instead of
  // using a "generic" CStyleCastNode.
  return std::make_unique<CStyleCastNode>(
      expr->location(),
      expr->result_type_deref().GetArrayElementType(nullptr).GetPointerType(),
      std::move(expr), CStyleCastKind::kPointer);
}

static CompilerType DoIntegralPromotion(
    std::shared_ptr<ExecutionContextScope> ctx, CompilerType from) {
  assert((IsInteger(from) || IsUnscopedEnum(from)) &&
         "Integral promotion works only for integers and unscoped enums.");

  // Don't do anything if the type doesn't need to be promoted.
  if (!IsPromotableIntegerType(from)) {
    return from;
  }

  if (IsUnscopedEnum(from)) {
    // Get the enumeration underlying type and promote it.
    return DoIntegralPromotion(ctx, GetEnumerationIntegerType(from));
  }

  // At this point the type should an integer.
  assert(IsInteger(from) && "invalid type: must be an integer");

  // Get the underlying builtin representation.
  lldb::BasicType builtin_type =
      from.GetCanonicalType().GetBasicTypeEnumeration();

  uint64_t from_size = 0;
  if (builtin_type == lldb::eBasicTypeWChar ||
      builtin_type == lldb::eBasicTypeSignedWChar ||
      builtin_type == lldb::eBasicTypeUnsignedWChar ||
      builtin_type == lldb::eBasicTypeChar16 ||
      builtin_type == lldb::eBasicTypeChar32) {
    // Find the type that can hold the entire range of values for our type.
    bool is_signed = IsSigned(from);
    if (auto temp = from.GetByteSize(ctx.get()))
      from_size = temp.value();

    CompilerType promote_types[] = {
      GetBasicType(ctx, lldb::eBasicTypeInt),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedInt),
      GetBasicType(ctx, lldb::eBasicTypeLong),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedLong),
      GetBasicType(ctx, lldb::eBasicTypeLongLong),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedLongLong),
    };
    for (auto& type : promote_types) {
      uint64_t byte_size = 0;
      if (auto temp = type.GetByteSize(ctx.get()))
        byte_size = temp.value();
      if (from_size < byte_size ||
          (from_size == byte_size &&
           is_signed ==(bool)(
               type.GetTypeInfo() & lldb::eTypeIsSigned)))
      {
        return type;
      }
    }

    llvm_unreachable("char type should fit into long long");
  }

  // Here we can promote only to "int" or "unsigned int".
  CompilerType int_type = GetBasicType(ctx, lldb::eBasicTypeInt);
  uint64_t int_byte_size = 0;
  if (auto temp = int_type.GetByteSize(ctx.get()))
    int_byte_size = temp.value();

  // Signed integer types can be safely promoted to "int".
  if (IsSigned(from)) {
    return int_type;
  }
  // Unsigned integer types are promoted to "unsigned int" if "int" cannot hold
  // their entire value range.
  return (from_size == int_byte_size)
      ? GetBasicType(ctx, lldb::eBasicTypeUnsignedInt)
      : int_type;
}

static ExprResult UsualUnaryConversions(
    std::shared_ptr<ExecutionContextScope>  ctx, ExprResult expr) {
  // Perform usual conversions for unary operators. At the moment this includes
  // array-to-pointer and the integral promotion for eligible types.
  auto result_type = expr->result_type_deref();

  if (expr->is_bitfield()) {
    // Promote bitfields. If `int` can represent the bitfield value, it is
    // converted to `int`. Otherwise, if `unsigned int` can represent it, it
    // is converted to `unsigned int`. Otherwise, it is treated as its
    // underlying type.

    uint32_t bitfield_size = expr->bitfield_size();
    // Some bitfields have undefined size (e.g. result of ternary operation).
    // The AST's `bitfield_size` of those is 0, and no promotion takes place.
    if (bitfield_size > 0 && IsInteger(result_type)) {
      auto int_type = GetBasicType(ctx, lldb::eBasicTypeInt);
      auto uint_type = GetBasicType(ctx, lldb::eBasicTypeUnsignedInt);
      uint64_t int_byte_size = 0;
      uint64_t uint_byte_size = 0;
      if (auto temp = int_type.GetByteSize(ctx.get()))
        int_byte_size = temp.value();
      if (auto temp = uint_type.GetByteSize(ctx.get()))
        uint_byte_size = temp.value();
      uint32_t int_bit_size = int_byte_size * CHAR_BIT;
      if (bitfield_size < int_bit_size ||
          (IsSigned(result_type) && bitfield_size == int_bit_size)) {
        expr = std::make_unique<CStyleCastNode>(expr->location(), int_type,
                                                std::move(expr),
                                                CStyleCastKind::kArithmetic);
      } else if (bitfield_size <= uint_byte_size * CHAR_BIT) {
        expr = std::make_unique<CStyleCastNode>(expr->location(), uint_type,
                                                std::move(expr),
                                                CStyleCastKind::kArithmetic);
      }
    }
  }

  if (IsArrayType(result_type)) {
    expr = InsertArrayToPointerConversion(std::move(expr));
  }

  if (IsInteger(result_type) || IsUnscopedEnum(result_type)) {
    auto promoted_type = DoIntegralPromotion(ctx, result_type);

    // Insert a cast if the type promotion is happening.
    // TODO(werat): Make this an implicit static_cast.
    if (!CompareTypes(promoted_type, result_type)) {
      expr = std::make_unique<CStyleCastNode>(expr->location(), promoted_type,
                                              std::move(expr),
                                              CStyleCastKind::kArithmetic);
    }
  }

  return expr;
}

static size_t ConversionRank(CompilerType type) {
  // Get integer conversion rank
  // https://eel.is/c++draft/conv.rank
  switch (type.GetCanonicalType().GetBasicTypeEnumeration()) {
    case lldb::eBasicTypeBool:
      return 1;
    case lldb::eBasicTypeChar:
    case lldb::eBasicTypeSignedChar:
    case lldb::eBasicTypeUnsignedChar:
      return 2;
    case lldb::eBasicTypeShort:
    case lldb::eBasicTypeUnsignedShort:
      return 3;
    case lldb::eBasicTypeInt:
    case lldb::eBasicTypeUnsignedInt:
      return 4;
    case lldb::eBasicTypeLong:
    case lldb::eBasicTypeUnsignedLong:
      return 5;
    case lldb::eBasicTypeLongLong:
    case lldb::eBasicTypeUnsignedLongLong:
      return 6;

      // TODO: The ranks of char16_t, char32_t, and wchar_t are equal to the
      // ranks of their underlying types.
    case lldb::eBasicTypeWChar:
    case lldb::eBasicTypeSignedWChar:
    case lldb::eBasicTypeUnsignedWChar:
      return 3;
    case lldb::eBasicTypeChar16:
      return 3;
    case lldb::eBasicTypeChar32:
      return 4;

    default:
      break;
  }
  return 0;
}

static lldb::BasicType BasicTypeToUnsigned(lldb::BasicType basic_type) {
  switch (basic_type) {
    case lldb::eBasicTypeInt:
      return lldb::eBasicTypeUnsignedInt;
    case lldb::eBasicTypeLong:
      return lldb::eBasicTypeUnsignedLong;
    case lldb::eBasicTypeLongLong:
      return lldb::eBasicTypeUnsignedLongLong;
    default:
      return basic_type;
  }
}

static void PerformIntegerConversions(std::shared_ptr<ExecutionContextScope> ctx,
                                      ExprResult& l,
                                      ExprResult& r, bool convert_lhs,
                                      bool convert_rhs) {
  // Assert that rank(l) < rank(r).
  auto l_type = l->result_type_deref();
  auto r_type = r->result_type_deref();

  // if `r` is signed and `l` is unsigned, check whether it can represent all
  // of the values of the type of the `l`. If not, then promote `r` to the
  // unsigned version of its type.
  if (IsSigned(r_type) && !IsSigned(l_type)) {
    uint64_t l_size = 0;
    uint64_t r_size = 0;
    if (auto temp = l_type.GetByteSize(ctx.get()))
      l_size = temp.value();;
    if (auto temp = r_type.GetByteSize(ctx.get()))
      r_size = temp.value();

    assert(l_size <= r_size && "left value must not be larger then the right!");

    if (r_size == l_size) {
      auto r_type_unsigned = GetBasicType(
          ctx,
          BasicTypeToUnsigned(r_type.GetCanonicalType()
                                  .GetBasicTypeEnumeration()));
      if (convert_rhs) {
        r = std::make_unique<CStyleCastNode>(r->location(), r_type_unsigned,
                                             std::move(r),
                                             CStyleCastKind::kArithmetic);
      }
    }
  }

  if (convert_lhs) {
    l = std::make_unique<CStyleCastNode>(l->location(), r->result_type(),
                                         std::move(l),
                                         CStyleCastKind::kArithmetic);
  }
}

static CompilerType UsualArithmeticConversions(
    std::shared_ptr<ExecutionContextScope> ctx,
    ExprResult& lhs,
    ExprResult& rhs,
    bool is_comp_assign = false) {
  // Apply unary conversions (e.g. intergal promotion) for both operands.
  // In case of a composite assignment operator LHS shouldn't get promoted.
  if (!is_comp_assign) {
    lhs = UsualUnaryConversions(ctx, std::move(lhs));
  }
  rhs = UsualUnaryConversions(ctx, std::move(rhs));

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  if (CompareTypes(lhs_type, rhs_type)) {
    return lhs_type;
  }

  // If either of the operands is not arithmetic (e.g. pointer), we're done.
  if (!IsScalar(lhs_type) || !IsScalar(rhs_type)) {
    CompilerType bad_type;
    return bad_type;
  }

  // Handle conversions for floating types (float, double).
  if (IsFloat(lhs_type) || IsFloat(rhs_type)) {
    // If both are floats, convert the smaller operand to the bigger.
    if (IsFloat(lhs_type) && IsFloat(rhs_type)) {
      int order = lhs_type.GetBasicTypeEnumeration() -
                  rhs_type.GetBasicTypeEnumeration();
      if (order > 0) {
        rhs = std::make_unique<CStyleCastNode>(rhs->location(), lhs_type,
                                               std::move(rhs),
                                               CStyleCastKind::kArithmetic);
        return lhs_type;
      }
      assert(order < 0 && "illegal operands: must not be of the same type");
      if (!is_comp_assign) {
        lhs = std::make_unique<CStyleCastNode>(lhs->location(), rhs_type,
                                               std::move(lhs),
                                               CStyleCastKind::kArithmetic);
      }
      return rhs_type;
    }

    if (IsFloat(lhs_type)) {
      assert(IsInteger(rhs_type) && "illegal operand: must be an integer");
      rhs = std::make_unique<CStyleCastNode>(rhs->location(), lhs_type,
                                             std::move(rhs),
                                             CStyleCastKind::kArithmetic);
      return lhs_type;
    }
    assert(IsFloat(rhs_type) && "illegal operand: must be a float");
    if (!is_comp_assign) {
      lhs = std::make_unique<CStyleCastNode>(lhs->location(), rhs_type,
                                             std::move(lhs),
                                             CStyleCastKind::kArithmetic);
    }
    return rhs_type;
  }

  // Handle conversion for integer types.
  assert((IsInteger(lhs_type) && IsInteger(rhs_type)) &&
         "illegal operands: must be both integers");

  using Rank = std::tuple<size_t, bool>;
  Rank l_rank = {ConversionRank(lhs_type), !IsSigned(lhs_type)};
  Rank r_rank = {ConversionRank(rhs_type), !IsSigned(rhs_type)};

  if (l_rank < r_rank) {
    PerformIntegerConversions(ctx, lhs, rhs, !is_comp_assign, true);
  } else if (l_rank > r_rank) {
    PerformIntegerConversions(ctx, rhs, lhs, true, !is_comp_assign);
  }

  if (!is_comp_assign) {
    assert(CompareTypes(lhs->result_type_deref(), rhs->result_type_deref()) &&
           "integral promotion error: operands result types must be the same");
  }

  return lhs->result_type_deref().GetCanonicalType();
}

static TypeDeclaration::TypeSpecifier ToTypeSpecifier(
    clang::tok::TokenKind kind) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  switch (kind) {
      // clang-format off
    case clang::tok::kw_void:     return TypeSpecifier::kVoid;
    case clang::tok::kw_bool:     return TypeSpecifier::kBool;
    case clang::tok::kw_char:     return TypeSpecifier::kChar;
    case clang::tok::kw_short:    return TypeSpecifier::kShort;
    case clang::tok::kw_int:      return TypeSpecifier::kInt;
    case clang::tok::kw_long:     return TypeSpecifier::kLong;
    case clang::tok::kw_float:    return TypeSpecifier::kFloat;
    case clang::tok::kw_double:   return TypeSpecifier::kDouble;
    case clang::tok::kw_wchar_t:  return TypeSpecifier::kWChar;
    case clang::tok::kw_char16_t: return TypeSpecifier::kChar16;
    case clang::tok::kw_char32_t: return TypeSpecifier::kChar32;
      // clang-format on
    default:
      assert(false && "invalid type specifier token");
      return TypeSpecifier::kUnknown;
  }
}

std::tuple<lldb::BasicType, bool> PickIntegerType(
    std::shared_ptr<ExecutionContextScope> ctx,
    const clang::NumericLiteralParser& literal,
    const llvm::APInt& value) {
  uint64_t int_byte_size = 0;
  uint64_t long_byte_size = 0;
  uint64_t long_long_byte_size = 0;
  if (auto temp = GetBasicType(ctx, lldb::eBasicTypeInt).GetByteSize(nullptr))
    int_byte_size = temp.value();

  if (auto temp = GetBasicType(ctx, lldb::eBasicTypeLong).GetByteSize(nullptr))
    long_byte_size = temp.value();

  if (auto temp = GetBasicType(ctx,
                               lldb::eBasicTypeLongLong).GetByteSize(nullptr))
    long_long_byte_size = temp.value();

  unsigned int_size = int_byte_size * CHAR_BIT;
  unsigned long_size = long_byte_size * CHAR_BIT;
  unsigned long_long_size = long_long_byte_size * CHAR_BIT;


  // Binary, Octal, Hexadecimal and literals with a U suffix are allowed to be
  // an unsigned integer.
  bool unsigned_is_allowed = literal.isUnsigned || literal.getRadix() != 10;

  // Try int/unsigned int.
  if (!literal.isLong && !literal.isLongLong && value.isIntN(int_size)) {
    if (!literal.isUnsigned && value.isIntN(int_size - 1)) {
      return {lldb::eBasicTypeInt, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedInt, true};
    }
  }
  // Try long/unsigned long.
  if (!literal.isLongLong && value.isIntN(long_size)) {
    if (!literal.isUnsigned && value.isIntN(long_size - 1)) {
      return {lldb::eBasicTypeLong, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedLong, true};
    }
  }
  // Try long long/unsigned long long.
  if (value.isIntN(long_long_size)) {
    if (!literal.isUnsigned && value.isIntN(long_long_size - 1)) {
      return {lldb::eBasicTypeLongLong, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedLongLong, true};
    }
  }

  // If we still couldn't decide a type, we probably have something that does
  // not fit in a signed long long, but has no U suffix. Also known as:
  //
  //  warning: integer literal is too large to be represented in a signed
  //  integer type, interpreting as unsigned [-Wimplicitly-unsigned-literal]
  //
  return {lldb::eBasicTypeUnsignedLongLong, true};
}

lldb::BasicType PickCharType(const clang::CharLiteralParser& literal) {
  if (literal.isMultiChar()) {
    return lldb::eBasicTypeInt;
#if LLVM_VERSION_MAJOR < 15
  } else if (literal.isAscii()) {
#else
  } else if (literal.isOrdinary()) {
#endif
    return lldb::eBasicTypeChar;
  } else if (literal.isWide()) {
    return lldb::eBasicTypeWChar;
  } else if (literal.isUTF8()) {
    // TODO: Change to eBasicTypeChar8 when support for u8 is added
    return lldb::eBasicTypeChar;
  } else if (literal.isUTF16()) {
    return lldb::eBasicTypeChar16;
  } else if (literal.isUTF32()) {
    return lldb::eBasicTypeChar32;
  }
  return lldb::eBasicTypeChar;
}

lldb::BasicType PickCharType(const clang::StringLiteralParser& literal) {
#if LLVM_VERSION_MAJOR < 15
  if (literal.isAscii()) {
#else
  if (literal.isOrdinary()) {
#endif
    return lldb::eBasicTypeChar;
  } else if (literal.isWide()) {
    return lldb::eBasicTypeWChar;
  } else if (literal.isUTF8()) {
    // TODO: Change to eBasicTypeChar8 when support for u8 is added.
    return lldb::eBasicTypeChar;
  } else if (literal.isUTF16()) {
    return lldb::eBasicTypeChar16;
  } else if (literal.isUTF32()) {
    return lldb::eBasicTypeChar32;
  }
  return lldb::eBasicTypeChar;
}

DILParser::DILParser(std::shared_ptr<DILSourceManager> dil_sm,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     bool use_synthetic)
    : m_sm(dil_sm), m_ctx_scope(exe_ctx_scope), m_use_synthetic(use_synthetic)
{
  clang::SourceManager& sm = dil_sm->GetSourceManager();;
  clang::DiagnosticsEngine& de = sm.getDiagnostics();

  auto tOpts = std::make_shared<clang::TargetOptions>();
  tOpts->Triple = llvm::sys::getDefaultTargetTriple();

  m_ti.reset(clang::TargetInfo::CreateTargetInfo(de, tOpts));

  m_lang_opts = std::make_unique<clang::LangOptions>();
  m_lang_opts->Bool = true;
  m_lang_opts->WChar = true;
  m_lang_opts->CPlusPlus = true;
  m_lang_opts->CPlusPlus11 = true;
  m_lang_opts->CPlusPlus14 = true;
  m_lang_opts->CPlusPlus17 = true;

  m_tml = std::make_unique<clang::TrivialModuleLoader>();

  auto hOpts = std::make_shared<clang::HeaderSearchOptions>();
  m_hs = std::make_unique<clang::HeaderSearch>(hOpts, sm, de, *m_lang_opts,
                                               m_ti.get());

  auto pOpts = std::make_shared<clang::PreprocessorOptions>();
  m_pp = std::make_unique<clang::Preprocessor>(pOpts, de, *m_lang_opts, sm,
                                               *m_hs, *m_tml);
  m_pp->Initialize(*m_ti);
  m_pp->EnterMainSourceFile();

  // Initialize the token.
  m_token.setKind(clang::tok::unknown);
}

ExprResult DILParser::Run(Status& error) {
  ConsumeToken();

  ExprResult expr;

  if (clang::tok::isStringLiteral(m_token.getKind()) &&
      m_pp->LookAhead(0).is(clang::tok::eof)) {
    // A special case to handle a single string-literal token.
    expr = ParseStringLiteral();
  } else {
    expr = ParseExpression();
  }


  Expect(clang::tok::eof);

  error = m_error;
  m_error.Clear();

  // Explicitly return DILErrorNode if there was an error during the parsing.
  // Some routines raise an error, but don't change the return value (e.g.
  // Expect).
  if (error.Fail()) {
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }
  return expr;
}

CompilerType DILParser::ResolveTypeDeclarators(
    CompilerType type, const std::vector<PtrOperator>& ptr_operators)
{
  CompilerType bad_type;
  // Resolve pointers/references.
  for (auto& [tk, loc] : ptr_operators) {
    if (tk == clang::tok::star) {
      // Pointers to reference types are forbidden.
      if (IsReferenceType(type)) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'type name' declared as a pointer to a "
                              "reference of type {0}",
                              TypeDescription(type)),
                loc);
        return bad_type;
      }
      // Get pointer type for the base type: e.g. int* -> int**.
      type = GetPointerType(type);

    } else if (tk == clang::tok::amp) {
      // References to references are forbidden.
      if (IsReferenceType(type)) {
        BailOut(ErrorCode::kInvalidOperandType,
                "type name declared as a reference to a reference", loc);
        return bad_type;
      }
      // Get reference type for the base type: e.g. int -> int&.
      type = GetReferenceType(type);
    }
  }

  return type;
}

bool DILParser::IsSimpleTypeSpecifierKeyword(clang::Token token) const {
  return token.isOneOf(
      clang::tok::kw_char, clang::tok::kw_char16_t, clang::tok::kw_char32_t,
      clang::tok::kw_wchar_t, clang::tok::kw_bool, clang::tok::kw_short,
      clang::tok::kw_int, clang::tok::kw_long, clang::tok::kw_signed,
      clang::tok::kw_unsigned, clang::tok::kw_float, clang::tok::kw_double,
      clang::tok::kw_void);
}

bool DILParser::IsCvQualifier(clang::Token token) const {
  return token.isOneOf(clang::tok::kw_const, clang::tok::kw_volatile);
}

bool DILParser::IsPtrOperator(clang::Token token) const {
  return token.isOneOf(clang::tok::star, clang::tok::amp);
}

bool DILParser::HandleSimpleTypeSpecifier(TypeDeclaration* type_decl) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  using SignSpecifier = TypeDeclaration::SignSpecifier;

  TypeSpecifier type_spec = type_decl->m_type_specifier;
  clang::SourceLocation loc = m_token.getLocation();
  clang::tok::TokenKind kind = m_token.getKind();

  switch (kind) {
    case clang::tok::kw_int: {
      // "int" can have signedness and be combined with "short", "long" and
      // "long long" (but not with another "int").
      if (type_decl->m_has_int_specifier) {
        BailOut(ErrorCode::kInvalidOperandType,
                "cannot combine with previous 'int' declaration specifier",
                loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kShort ||
          type_spec == TypeSpecifier::kLong ||
          type_spec == TypeSpecifier::kLongLong) {
        type_decl->m_has_int_specifier = true;
        return true;
      } else if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kInt;
        type_decl->m_has_int_specifier = true;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case clang::tok::kw_long: {
      // "long" can have signedness and be combined with "int" or "long" to
      // form "long long".
      if (type_spec == TypeSpecifier::kUnknown ||
          type_spec == TypeSpecifier::kInt) {
        type_decl->m_type_specifier = TypeSpecifier::kLong;
        return true;
      } else if (type_spec == TypeSpecifier::kLong) {
        type_decl->m_type_specifier = TypeSpecifier::kLongLong;
        return true;
      } else if (type_spec == TypeSpecifier::kDouble) {
        type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case clang::tok::kw_short: {
      // "short" can have signedness and be combined with "int".
      if (type_spec == TypeSpecifier::kUnknown ||
          type_spec == TypeSpecifier::kInt) {
        type_decl->m_type_specifier = TypeSpecifier::kShort;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case clang::tok::kw_char: {
      // "char" can have signedness, but it cannot be combined with any other
      // type specifier.
      if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kChar;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case clang::tok::kw_double: {
      // "double" can be combined with "long" to form "long double", but it
      // cannot be combined with signedness specifier.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                "'double' cannot be signed or unsigned", loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kDouble;
        return true;
      } else if (type_spec == TypeSpecifier::kLong) {
        type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case clang::tok::kw_bool:
    case clang::tok::kw_void:
    case clang::tok::kw_float:
    case clang::tok::kw_wchar_t:
    case clang::tok::kw_char16_t:
    case clang::tok::kw_char32_t: {
      // These types cannot have signedness or be combined with any other type
      // specifiers.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'{0}' cannot be signed or unsigned",
                              ToString(ToTypeSpecifier(kind))),
                loc);
        return false;
      }
      if (type_spec != TypeSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cannot combine with previous '{0}' declaration specifier",
                    ToString(type_spec)),
                loc);
      }
      type_decl->m_type_specifier = ToTypeSpecifier(kind);
      return true;
    }

    case clang::tok::kw_signed:
    case clang::tok::kw_unsigned: {
      // "signed" and "unsigned" cannot be combined with another signedness
      // specifier.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cannot combine with previous '{0}' declaration specifier",
                    ToString(type_decl->m_sign_specifier)),
                loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kVoid ||
          type_spec == TypeSpecifier::kBool ||
          type_spec == TypeSpecifier::kFloat ||
          type_spec == TypeSpecifier::kDouble ||
          type_spec == TypeSpecifier::kLongDouble ||
          type_spec == TypeSpecifier::kWChar ||
          type_spec == TypeSpecifier::kChar16 ||
          type_spec == TypeSpecifier::kChar32) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'{0}' cannot be signed or unsigned",
                              ToString(type_spec)),
                loc);
        return false;
      }

      type_decl->m_sign_specifier = (kind == clang::tok::kw_signed)
                                       ? SignSpecifier::kSigned
                                       : SignSpecifier::kUnsigned;
      return true;
    }

    default:
      assert(false && "invalid simple type specifier kind");
      return false;
  }
}

ExprResult DILParser::ParseStringLiteral() {
  ExpectOneOf(clang::tok::string_literal, clang::tok::wide_string_literal,
              clang::tok::utf8_string_literal, clang::tok::utf16_string_literal,
              clang::tok::utf32_string_literal);
  clang::SourceLocation loc = m_token.getLocation();

  // TODO: Support parsing of joined string-literals (e.g. "abc" "def").
  // Currently, only a single token can be parsed into a string.
  clang::StringLiteralParser string_literal(
      clang::ArrayRef<clang::Token>(m_token), *m_pp);

  if (string_literal.hadError) {
    // TODO: Use ErrorCode::kInvalidStringLiteral in the future.
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("Failed to parse token as string-literal: {0}",
                          TokenDescription(m_token)),
            loc);
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto char_type = GetBasicType(m_ctx_scope, PickCharType(string_literal));
  // Strings are terminated by a null value (add +1).
  CompilerType compiler_type = char_type;
  uint64_t byte_size = 0;
  if (auto temp = compiler_type.GetByteSize(nullptr))
    byte_size = temp.value();
  uint64_t array_size = string_literal.GetStringLength() / byte_size + 1;
  auto array_type = compiler_type.GetArrayType(array_size);

  clang::StringRef value = string_literal.GetString();
  std::vector<char> data(value.data(), value.data() + value.size());
  // Add the terminating null bytes.
  data.insert(data.end(), byte_size, 0);

  assert(data.size() == array_type.GetByteSize(nullptr) &&
         "invalid string literal: unexpected data size");
  ConsumeToken();
  return std::make_unique<LiteralNode>(loc, array_type,
                                       std::move(data), false);
}

// Parse an expression.
//
//  expression:
//    assignment_expression
//
ExprResult DILParser::ParseExpression() { return ParseAssignmentExpression(); }

// Parse an assingment_expression.
//
//  assignment_expression:
//    conditional_expression
//    logical_or_expression assignment_operator assignment_expression
//
//  assignment_operator:
//    "="
//    "*="
//    "/="
//    "%="
//    "+="
//    "-="
//    ">>="
//    "<<="
//    "&="
//    "^="
//    "|="
//
//  conditional_expression:
//    logical_or_expression
//    logical_or_expression "?" expression ":" assignment_expression
//
ExprResult DILParser::ParseAssignmentExpression() {
  auto lhs = ParseLogicalOrExpression();

  // Check if it's an assingment expression.
  if (m_token.isOneOf(clang::tok::equal, clang::tok::starequal,
                     clang::tok::slashequal, clang::tok::percentequal,
                     clang::tok::plusequal, clang::tok::minusequal,
                     clang::tok::greatergreaterequal, clang::tok::lesslessequal,
                     clang::tok::ampequal, clang::tok::caretequal,
                     clang::tok::pipeequal)) {
    // That's an assingment!
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseAssignmentExpression();
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  // Check if it's a conditional expression.
  if (m_token.is(clang::tok::question)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto true_val = ParseExpression();
    Expect(clang::tok::colon);
    ConsumeToken();
    auto false_val = ParseAssignmentExpression();
    lhs = BuildTernaryOp(std::move(lhs), std::move(true_val),
                         std::move(false_val), token.getLocation());
  }

  return lhs;
}

// Parse a logical_or_expression.
//
//  logical_or_expression:
//    logical_and_expression {"||" logical_and_expression}
//
ExprResult DILParser::ParseLogicalOrExpression() {
  auto lhs = ParseLogicalAndExpression();

  while (m_token.is(clang::tok::pipepipe)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseLogicalAndExpression();
    lhs = BuildBinaryOp(BinaryOpKind::LOr, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse a logical_and_expression.
//
//  logical_and_expression:
//    inclusive_or_expression {"&&" inclusive_or_expression}
//
ExprResult DILParser::ParseLogicalAndExpression() {
  auto lhs = ParseInclusiveOrExpression();

  while (m_token.is(clang::tok::ampamp)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseInclusiveOrExpression();
    lhs = BuildBinaryOp(BinaryOpKind::LAnd, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an inclusive_or_expression.
//
//  inclusive_or_expression:
//    exclusive_or_expression {"|" exclusive_or_expression}
//
ExprResult DILParser::ParseInclusiveOrExpression() {
  auto lhs = ParseExclusiveOrExpression();

  while (m_token.is(clang::tok::pipe)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseExclusiveOrExpression();
    lhs = BuildBinaryOp(BinaryOpKind::Or, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an exclusive_or_expression.
//
//  exclusive_or_expression:
//    and_expression {"^" and_expression}
//
ExprResult DILParser::ParseExclusiveOrExpression() {
  auto lhs = ParseAndExpression();

  while (m_token.is(clang::tok::caret)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseAndExpression();
    lhs = BuildBinaryOp(BinaryOpKind::Xor, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an and_expression.
//
//  and_expression:
//    equality_expression {"&" equality_expression}
//
ExprResult DILParser::ParseAndExpression() {
  auto lhs = ParseEqualityExpression();

  while (m_token.is(clang::tok::amp)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseEqualityExpression();
    lhs = BuildBinaryOp(BinaryOpKind::And, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an equality_expression.
//
//  equality_expression:
//    relational_expression {"==" relational_expression}
//    relational_expression {"!=" relational_expression}
//
ExprResult DILParser::ParseEqualityExpression() {
  auto lhs = ParseRelationalExpression();

  while (m_token.isOneOf(clang::tok::equalequal, clang::tok::exclaimequal)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseRelationalExpression();
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a relational_expression.
//
//  relational_expression:
//    shift_expression {"<" shift_expression}
//    shift_expression {">" shift_expression}
//    shift_expression {"<=" shift_expression}
//    shift_expression {">=" shift_expression}
//
ExprResult DILParser::ParseRelationalExpression() {
  auto lhs = ParseShiftExpression();

  while (m_token.isOneOf(clang::tok::less, clang::tok::greater,
                        clang::tok::lessequal, clang::tok::greaterequal)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseShiftExpression();
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a shift_expression.
//
//  shift_expression:
//    additive_expression {"<<" additive_expression}
//    additive_expression {">>" additive_expression}
//
ExprResult DILParser::ParseShiftExpression() {
  auto lhs = ParseAdditiveExpression();

  while (m_token.isOneOf(clang::tok::lessless, clang::tok::greatergreater)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseAdditiveExpression();
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse an additive_expression.
//
//  additive_expression:
//    multiplicative_expression {"+" multiplicative_expression}
//    multiplicative_expression {"-" multiplicative_expression}
//
ExprResult DILParser::ParseAdditiveExpression() {
  auto lhs = ParseMultiplicativeExpression();

  while (m_token.isOneOf(clang::tok::plus, clang::tok::minus)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseMultiplicativeExpression();
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a multiplicative_expression.
//
//  multiplicative_expression:
//    cast_expression {"*" cast_expression}
//    cast_expression {"/" cast_expression}
//    cast_expression {"%" cast_expression}
//
ExprResult DILParser::ParseMultiplicativeExpression() {
  auto lhs = ParseCastExpression();
  // auto lhs = ParseUnaryExpression(); // Question: Andy, is this right?

  while (m_token.isOneOf(clang::tok::star, clang::tok::slash,
                        clang::tok::percent)) {
    clang::Token token = m_token;
    ConsumeToken();
    auto rhs = ParseCastExpression();
    // auto rhs = ParseUnaryExpression(); // Question: Andy, is this right?
    lhs = BuildBinaryOp(clang_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}



// Parse a cast_expression.
//
//  cast_expression:
//    unary_expression
//    "(" type_id ")" cast_expression
//
ExprResult DILParser::ParseCastExpression() {
  // This can be a C-style cast, try parsing the contents as a type declaration.
  if (m_token.is(clang::tok::l_paren)) {
    clang::Token token = m_token;

    // Enable lexer backtracking, so that we can rollback in case it's not
    // actually a type declaration.
    TentativeParsingAction tentative_parsing(this);

    // Consume the token only after enabling the backtracking.
    ConsumeToken();

    // Try parsing the type declaration. If the returned value is not valid,
    // then we should rollback and try parsing the expression.
    auto type_id = ParseTypeId();
    if (type_id) {
      // Successfully parsed the type declaration. Commit the backtracked
      // tokens and parse the cast_expression.
      tentative_parsing.Commit();

      if (!type_id.value().IsValid()) {
        CompilerType bad_type;
        return std::make_unique<DILErrorNode>(bad_type);
      }

      Expect(clang::tok::r_paren);
      ConsumeToken();
      auto rhs = ParseCastExpression();

      return BuildCStyleCast(type_id.value(), std::move(rhs),
                             token.getLocation());
    }

    // Failed to parse the contents of the parentheses as a type declaration.
    // Rollback the lexer and try parsing it as unary_expression.
    tentative_parsing.Rollback();
  }

  return ParseUnaryExpression();
}


// Parse an unary_expression.
//
//  unary_expression:
//    postfix_expression
//    "++" cast_expression
//    "--" cast_expression
//    unary_operator cast_expression
//    sizeof unary_expression
//    sizeof "(" type_id ")"
//
//  unary_operator:
//    "&"
//    "*"
//    "+"
//    "-"
//    "~"
//    "!"
//
ExprResult DILParser::ParseUnaryExpression() {
  if (m_token.isOneOf(clang::tok::plusplus, clang::tok::minusminus,
                     clang::tok::star, clang::tok::amp, clang::tok::plus,
                     clang::tok::minus, clang::tok::exclaim,
                     clang::tok::tilde)) {
    clang::Token token = m_token;
    clang::SourceLocation loc = token.getLocation();
    ConsumeToken();
    auto rhs = ParseCastExpression();
    // auto rhs = ParseUnaryExpression();

    switch (token.getKind()) {
      case clang::tok::plusplus:
        return BuildUnaryOp(UnaryOpKind::PreInc, std::move(rhs), loc);
      case clang::tok::minusminus:
        return BuildUnaryOp(UnaryOpKind::PreDec, std::move(rhs), loc);
      case clang::tok::star:
        return BuildUnaryOp(UnaryOpKind::Deref, std::move(rhs), loc);
      case clang::tok::amp:
        return BuildUnaryOp(UnaryOpKind::AddrOf, std::move(rhs), loc);
      case clang::tok::plus:
        return BuildUnaryOp(UnaryOpKind::Plus, std::move(rhs), loc);
      case clang::tok::minus:
        return BuildUnaryOp(UnaryOpKind::Minus, std::move(rhs), loc);
      case clang::tok::tilde:
        return BuildUnaryOp(UnaryOpKind::Not, std::move(rhs), loc);
      case clang::tok::exclaim:
        return BuildUnaryOp(UnaryOpKind::LNot, std::move(rhs), loc);

      default:
        llvm_unreachable("invalid token kind");
    }
  }

  if (m_token.is(clang::tok::kw_sizeof)) {
    clang::SourceLocation sizeof_loc = m_token.getLocation();
    ConsumeToken();

    // [expr.sizeof](http://eel.is/c++draft/expr.sizeof#1)
    //
    // The operand is either an expression, which is an unevaluated operand,
    // or a parenthesized type-id.

    // Either operand itself (if it's a type_id), or an operand return type
    // (if it's an expression).
    CompilerType operand;

    // `(` can mean either a type_id or a parenthesized expression.
    if (m_token.is(clang::tok::l_paren)) {
      TentativeParsingAction tentative_parsing(this);

      Expect(clang::tok::l_paren);
      ConsumeToken();

      // Parse the type definition and resolve the type.
      auto type_id = ParseTypeId();
      if (type_id) {
        tentative_parsing.Commit();

        // type_id requires parentheses, so there must be a closing one.
        Expect(clang::tok::r_paren);
        ConsumeToken();

        operand = type_id.value();

      } else {
        tentative_parsing.Rollback();

        // Failed to parse type_id, fallback to parsing an unary_expression.
        operand = ParseUnaryExpression()->result_type_deref();
      }

    } else {
      // No opening parenthesis means this must be an unary_expression.
      operand = ParseUnaryExpression()->result_type_deref();
    }
    if (!operand.IsValid()) {
      CompilerType bad_type;
      return std::make_unique<DILErrorNode>(bad_type);
    }

    lldb::BasicType size_type;
    llvm::Triple triple(llvm::Twine(
        m_ctx_scope->CalculateTarget()->GetArchitecture().GetTriple().str()));
    if (triple.isOSWindows()) {
      size_type = triple.isArch64Bit() ? lldb::eBasicTypeUnsignedLongLong
          : lldb::eBasicTypeUnsignedInt;
    } else {
      size_type = triple.isArch64Bit() ? lldb::eBasicTypeUnsignedLong
          : lldb::eBasicTypeUnsignedInt;
  }

    auto result_type = GetBasicType(m_ctx_scope, size_type);
    return std::make_unique<SizeOfNode>(sizeof_loc, result_type, operand);
  }

  return ParsePostfixExpression();
}

// Parse a postfix_expression.
//
//  postfix_expression:
//    primary_expression
//    postfix_expression "[" expression "]"
//    postfix_expression "." id_expression
//    postfix_expression "->" id_expression
//    postfix_expression "++"
//    postfix_expression "--"
//    static_cast "<" type_id ">" "(" expression ")" ;
//    dynamic_cast "<" type_id ">" "(" expression ")" ;
//    reinterpret_cast "<" type_id ">" "(" expression ")" ;
//
ExprResult DILParser::ParsePostfixExpression() {
  // Parse the first part of the postfix_expression. This could be either a
  // primary_expression, or a postfix_expression itself.
  ExprResult lhs;
  CompilerType bad_type;

  // C++-style cast.
  if (m_token.isOneOf(clang::tok::kw_static_cast, clang::tok::kw_dynamic_cast,
                     clang::tok::kw_reinterpret_cast)) {
    clang::tok::TokenKind cast_kind = m_token.getKind();
    clang::SourceLocation cast_loc = m_token.getLocation();
    ConsumeToken();

    Expect(clang::tok::less);
    ConsumeToken();

    clang::SourceLocation loc = m_token.getLocation();

    // Parse the type definition and resolve the type.
    auto type_id = ParseTypeId(/*must_be_type_id*/ true);
    if (!type_id) {
      BailOut(ErrorCode::kInvalidOperandType,
              "type name requires a specifier or qualifier", loc);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    if (!type_id.value().IsValid()) {
      return std::make_unique<DILErrorNode>(bad_type);
    }

    Expect(clang::tok::greater);
    ConsumeToken();

    Expect(clang::tok::l_paren);
    ConsumeToken();
    auto rhs = ParseExpression();
    Expect(clang::tok::r_paren);
    ConsumeToken();

    lhs = BuildCxxCast(cast_kind, type_id.value(), std::move(rhs), cast_loc);

  } else {
    // Otherwise it's a primary_expression.
    lhs = ParsePrimaryExpression();
  }
  assert(lhs && "LHS of the postfix_expression can't be NULL.");

  while (m_token.isOneOf(clang::tok::l_square, clang::tok::period,
                        clang::tok::arrow, clang::tok::plusplus,
                        clang::tok::minusminus)) {
    clang::Token token = m_token;
    switch (token.getKind()) {
      case clang::tok::period:
      case clang::tok::arrow: {
        ConsumeToken();
        clang::Token member_token = m_token;
        auto member_id = ParseIdExpression();
        // Check if this is a function call.
        if (m_token.is(clang::tok::l_paren)) {
          // TODO: Check if `member_id` is actually a member function of `lhs`.
          // If not, produce a more accurate diagnostic.
          BailOut(ErrorCode::kNotImplemented,
                  "member function calls are not supported",
                  m_token.getLocation());
        }
        lhs = BuildMemberOf(std::move(lhs), std::move(member_id),
                            token.getKind() == clang::tok::arrow,
                            member_token.getLocation());
        break;
      }
      case clang::tok::plusplus: {
        ConsumeToken();
        return BuildUnaryOp(UnaryOpKind::PostInc, std::move(lhs),
                            token.getLocation());
      }
      case clang::tok::minusminus: {
        ConsumeToken();
        return BuildUnaryOp(UnaryOpKind::PostDec, std::move(lhs),
                            token.getLocation());
      }
      case clang::tok::l_square: {
        ConsumeToken();
        auto rhs = ParseExpression();
        Expect(clang::tok::r_square);
        ConsumeToken();
        lhs = BuildBinarySubscript(std::move(lhs), std::move(rhs),
                                   token.getLocation());
        break;
      }

      default:
        llvm_unreachable("invalid token");
    }
  }

  return lhs;
}

// Parse a primary_expression.
//
//  primary_expression:
//    numeric_literal
//    boolean_literal
//    pointer_literal
//    id_expression
//    "this"
//    "(" expression ")"
//    builtin_func
//
ExprResult DILParser::ParsePrimaryExpression() {
  CompilerType bad_type;
  if (m_token.is(clang::tok::numeric_constant)) {
    return ParseNumericLiteral();
  } else if (m_token.isOneOf(clang::tok::kw_true, clang::tok::kw_false)) {
    return ParseBooleanLiteral();
  } else if (m_token.isOneOf(clang::tok::char_constant,
                            clang::tok::wide_char_constant,
                            clang::tok::utf8_char_constant,
                            clang::tok::utf16_char_constant,
                            clang::tok::utf32_char_constant)) {
    return ParseCharLiteral();
  } else if (clang::tok::isStringLiteral(m_token.getKind())) {
    // Note: Only expressions that consist of a single string literal can be
    // handled by DIL.
    BailOut(ErrorCode::kNotImplemented, "string literals are not supported",
            m_token.getLocation());
    return std::make_unique<DILErrorNode>(bad_type);
  } else if (m_token.is(clang::tok::kw_nullptr)) {
    return ParsePointerLiteral();
  } else if (m_token.isOneOf(clang::tok::coloncolon, clang::tok::identifier)) {
    // Save the source location for the diagnostics message.
    clang::SourceLocation loc = m_token.getLocation();
    auto identifier = ParseIdExpression();
    // Check if this is a function call.
    if (m_token.is(clang::tok::l_paren)) {
      auto func_def = GetBuiltinFunctionDef(m_ctx_scope, identifier);
      if (!func_def) {
        BailOut(
            ErrorCode::kNotImplemented,
            llvm::formatv("function '{0}' is not a supported builtin intrinsic",
                          identifier),
            loc);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      return ParseBuiltinFunction(loc, std::move(func_def));
    }
    // Otherwise look for an identifier.
    // TODO: Handle bitfield identifiers when evaluating in the value context.
    auto value = LookupIdentifier(identifier, m_ctx_scope);
    if (!value->IsValid()) {
      BailOut(ErrorCode::kUndeclaredIdentifier,
              llvm::formatv("use of undeclared identifier '{0}'", identifier),
              loc);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    return std::make_unique<IdentifierNode>(loc, identifier, std::move(value),
                                            /*is_rvalue*/ false,
                                            IsContextVar(identifier));
  } else if (m_token.is(clang::tok::kw_this)) {
    // Save the source location for the diagnostics message.
    clang::SourceLocation loc = m_token.getLocation();
    ConsumeToken();
    auto value = LookupIdentifier("this", m_ctx_scope);
    if (!value->IsValid()) {
      BailOut(ErrorCode::kUndeclaredIdentifier,
              "invalid use of 'this' outside of a non-static member function",
              loc);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    // Special case for "this" pointer. As per C++ standard, it's a prvalue.
    return std::make_unique<IdentifierNode>(loc, "this", std::move(value),
                                            /*is_rvalue*/ true,
                                            /*is_context_var*/ false);
  } else if (m_token.is(clang::tok::l_paren)) {
    // Check in case this is an anonynmous namespace
    if (m_pp->LookAhead(0).is(clang::tok::identifier)
        && (m_pp->getSpelling(m_pp->LookAhead(0)) == "anonymous")
        && m_pp->LookAhead(1).is(clang::tok::kw_namespace)
        && m_pp->LookAhead(2).is(clang::tok::r_paren)
        && m_pp->LookAhead(3).is(clang::tok::coloncolon)) {
      ConsumeToken(); // l_paren
      ConsumeToken(); // identifier 'anonymous'
      ConsumeToken(); // keyword 'namespace'
      ConsumeToken(); // r_paren
      std::string identifier = "(anonymous namespace)";
      Expect(clang::tok::coloncolon);
      // Save the source location for the diagnostics message.
      clang::SourceLocation loc = m_token.getLocation();
      ConsumeToken();
      assert ((m_token.is(clang::tok::identifier) ||
               m_token.is(clang::tok::l_paren)) &&
              "Expected an identifier or anonymous namespeace, but not found.");
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        // There was only an identifer, no more levels of nesting. Or there
        // was an invalid expression starting with a left parenthesis.
        Expect(clang::tok::identifier);
        identifier2 = m_pp->getSpelling(m_token);
        ConsumeToken();
      }
      identifier = identifier + "::" + identifier2;
      auto value = LookupIdentifier(identifier, m_ctx_scope);
      if (!value->IsValid()) {
        BailOut(ErrorCode::kUndeclaredIdentifier,
                llvm::formatv("use of undeclared identifier '{0}'", identifier),
                loc);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      return std::make_unique<IdentifierNode>(loc, identifier, std::move(value),
                                              /*is_rvalue*/ false,
                                              IsContextVar(identifier));
    } else {
      ConsumeToken();
      auto expr = ParseExpression();
      Expect(clang::tok::r_paren);
      ConsumeToken();
      return expr;
    }
  }

  BailOut(ErrorCode::kInvalidExpressionSyntax,
          llvm::formatv("Unexpected token: {0}", TokenDescription(m_token)),
          m_token.getLocation());
  return std::make_unique<DILErrorNode>(bad_type);
}

// Parse a type_id.
//
//  type_id:
//    type_specifier_seq [abstract_declarator]
//
std::optional<CompilerType> DILParser::ParseTypeId(bool must_be_type_id) {
  clang::SourceLocation type_loc = m_token.getLocation();
  TypeDeclaration type_decl;
  CompilerType bad_type;

  // type_specifier_seq is required here, start with trying to parse it.
  ParseTypeSpecifierSeq(&type_decl);

  if (type_decl.IsEmpty()) {
    // TODO: Should we bail out if `must_be_type_id` is set?
    return {};
  }

  if (type_decl.m_has_error) {
    if (type_decl.m_is_builtin) {
      return bad_type;
    }

    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    // Found something looking like a user type, but failed to parse it.
    // Return invalid type if we expect to have a type here, otherwise nullopt.
    if (must_be_type_id) {
      return bad_type;
    }
    return {};
  }

  // Try to resolve the base type.
  CompilerType type;
  if (type_decl.m_is_builtin) {
    type = GetBasicType(m_ctx_scope, type_decl.GetBasicType());
    assert(type.IsValid() && "cannot resolve basic type");

  } else {
    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    type = ResolveTypeByName(type_decl.m_user_typename, m_ctx_scope);
    if (!type.IsValid()) {
      if (must_be_type_id) {
        BailOut(
            ErrorCode::kUndeclaredIdentifier,
            llvm::formatv("unknown type name '{0}'",
                          type_decl.m_user_typename),
            type_loc);
        return bad_type;
      }
      return {};
    }

    if (LookupIdentifier(type_decl.m_user_typename, m_ctx_scope)->IsValid()) {
      // Same-name identifiers should be preferred over typenames.
      // TODO: Make type accessible with 'class', 'struct' and 'union' keywords.
      if (must_be_type_id) {
        BailOut(ErrorCode::kUndeclaredIdentifier,
                llvm::formatv(
                    "must use '{0}' tag to refer to type '{1}' in this scope",
                    GetTypeTag(type), type_decl.m_user_typename),
                type_loc);
        return bad_type;
      }
      return {};
    }
  }

  //
  //  abstract_declarator:
  //    ptr_operator [abstract_declarator]
  //
  std::vector<DILParser::PtrOperator> ptr_operators;
  while (IsPtrOperator(m_token)) {
    ptr_operators.push_back(ParsePtrOperator());
  }
  type = ResolveTypeDeclarators(type, ptr_operators);

  return type;
}

// Parse a type_specifier_seq.
//
//  type_specifier_seq:
//    type_specifier [type_specifier_seq]
//
void DILParser::ParseTypeSpecifierSeq(TypeDeclaration* type_decl) {
  while (true) {
    bool type_specifier = ParseTypeSpecifier(type_decl);
    if (!type_specifier) {
      break;
    }
  }
}

// Parse a type_specifier.
//
//  type_specifier:
//    simple_type_specifier
//    cv_qualifier
//
//  simple_type_specifier:
//    ["::"] [nested_name_specifier] type_name
//    "char"
//    "char16_t"
//    "char32_t"
//    "wchar_t"
//    "bool"
//    "short"
//    "int"
//    "long"
//    "signed"
//    "unsigned"
//    "float"
//    "double"
//    "void"
//
// Returns TRUE if a type_specifier was successfully parsed at this location.
//
bool DILParser::ParseTypeSpecifier(TypeDeclaration* type_decl) {
  if (IsCvQualifier(m_token)) {
    // Just ignore CV quialifiers, we don't use them in type casting.
    ConsumeToken();
    return true;
  }

  if (IsSimpleTypeSpecifierKeyword(m_token)) {
    // User-defined typenames can't be combined with builtin keywords.
    if (type_decl->m_is_user_type) {
      BailOut(ErrorCode::kInvalidOperandType,
              "cannot combine with previous declaration specifier",
              m_token.getLocation());
      type_decl->m_has_error = true;
      return false;
    }

    // From now on this type declaration must describe a builtin type.
    // TODO: Should this be allowed -- `unsigned myint`?
    type_decl->m_is_builtin = true;

    if (!HandleSimpleTypeSpecifier(type_decl)) {
      type_decl->m_has_error = true;
      return false;
    }
    ConsumeToken();
    return true;
  }

  // The type_specifier must be a user-defined type. Try parsing a
  // simple_type_specifier.
  {
    // Try parsing optional global scope operator.
    bool global_scope = false;
    if (m_token.is(clang::tok::coloncolon)) {
      global_scope = true;
      ConsumeToken();
    }

    clang::SourceLocation loc = m_token.getLocation();

    // Try parsing optional nested_name_specifier.
    auto nested_name_specifier = ParseNestedNameSpecifier();

    // Try parsing required type_name.
    auto type_name = ParseTypeName();

    // If there is a type_name, then this is indeed a simple_type_specifier.
    // Global and qualified (namespace/class) scopes can be empty, since they're
    // optional. In this case type_name is type we're looking for.
    if (!type_name.empty()) {
      // User-defined typenames can't be combined with builtin keywords.
      if (type_decl->m_is_builtin) {
        BailOut(ErrorCode::kInvalidOperandType,
                "cannot combine with previous declaration specifier", loc);
        type_decl->m_has_error = true;
        return false;
      }
      // There should be only one user-defined typename.
      if (type_decl->m_is_user_type) {
        BailOut(ErrorCode::kInvalidOperandType,
                "two or more data types in declaration of 'type name'", loc);
        type_decl->m_has_error = true;
        return false;
      }

      // Construct the fully qualified typename.
      type_decl->m_is_user_type = true;
      type_decl->m_user_typename =
          llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                        nested_name_specifier, type_name);
      return true;
    }
  }

  // No type_specifier was found here.
  return false;
}

// Parse an id_expression.
//
//  id_expression:
//    unqualified_id
//    qualified_id
//
//  qualified_id:
//    ["::"] [nested_name_specifier] unqualified_id
//    ["::"] identifier
//
//  identifier:
//    ? clang::tok::identifier ?
//
std::string DILParser::ParseIdExpression() {
  // Try parsing optional global scope operator.
  bool global_scope = false;
  if (m_token.is(clang::tok::coloncolon)) {
    global_scope = true;
    ConsumeToken();
  }

  // Try parsing optional nested_name_specifier.
  auto nested_name_specifier = ParseNestedNameSpecifier();

  // If nested_name_specifier is present, then it's qualified_id production.
  // Follow the first production rule.
  if (!nested_name_specifier.empty()) {
    // Parse unqualified_id and construct a fully qualified id expression.
    auto unqualified_id = ParseUnqualifiedId();

    return llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                         nested_name_specifier, unqualified_id);
  }

  // No nested_name_specifier, but with global scope -- this is also a
  // qualified_id production. Follow the second production rule.
  else if (global_scope) {
    Expect(clang::tok::identifier);
    std::string identifier = m_pp->getSpelling(m_token);
    ConsumeToken();
    return llvm::formatv("{0}{1}", global_scope ? "::" : "", identifier);
  }

  // This is unqualified_id production.
  return ParseUnqualifiedId();
}

// Parse an unqualified_id.
//
//  unqualified_id:
//    identifier
//
//  identifier:
//    ? clang::tok::identifier ?
//
std::string DILParser::ParseUnqualifiedId() {
  Expect(clang::tok::identifier);
  std::string identifier = m_pp->getSpelling(m_token);
  ConsumeToken();
  return identifier;
}

// Parse a numeric_literal.
//
//  numeric_literal:
//    ? clang::tok::numeric_constant ?
//
ExprResult DILParser::ParseNumericLiteral() {
  Expect(clang::tok::numeric_constant);
  ExprResult numeric_constant = ParseNumericConstant(m_token);
  ConsumeToken();
  return numeric_constant;
}

// Parse an boolean_literal.
//
//  boolean_literal:
//    "true"
//    "false"
//
ExprResult DILParser::ParseBooleanLiteral() {
  ExpectOneOf(clang::tok::kw_true, clang::tok::kw_false);
  clang::SourceLocation loc = m_token.getLocation();
  bool literal_value = m_token.is(clang::tok::kw_true);
  ConsumeToken();
  return std::make_unique<LiteralNode>(
      loc, GetBasicType(m_ctx_scope, lldb::eBasicTypeBool), literal_value,
      /*is_literal_zero*/ false);
}

ExprResult DILParser::ParseCharLiteral() {
  ExpectOneOf(clang::tok::char_constant, clang::tok::wide_char_constant,
              clang::tok::utf8_char_constant, clang::tok::utf16_char_constant,
              clang::tok::utf32_char_constant);
  clang::SourceLocation loc = m_token.getLocation();

  std::string token_spelling = m_pp->getSpelling(m_token);

  const char* token_begin = token_spelling.c_str();
  clang::CharLiteralParser char_literal(token_begin,
                                        token_begin + token_spelling.size(),
                                        loc, *m_pp, m_token.getKind());

  if (char_literal.hadError()) {
    // TODO: Add new ErrorCode kInvalidCharLiteral and use it
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("Failed to parse token as char-constant: {0}",
                          TokenDescription(m_token)),
            m_token.getLocation());
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto ctx_basic_type = GetBasicType(m_ctx_scope, PickCharType(char_literal));
  uint64_t byte_size = 0;
  if (auto temp = ctx_basic_type.GetByteSize(nullptr))
    byte_size = temp.value();
  llvm::APInt literal_value(byte_size * CHAR_BIT,
                            char_literal.getValue());

  ConsumeToken();
  return std::make_unique<LiteralNode>(loc, ctx_basic_type, literal_value,
                                       /*is_literal_zero*/ false);
}

// Parse an pointer_literal.
//
//  pointer_literal:
//    "nullptr"
//
ExprResult DILParser::ParsePointerLiteral() {
  Expect(clang::tok::kw_nullptr);
  clang::SourceLocation loc = m_token.getLocation();
  ConsumeToken();
  llvm::APInt raw_value(type_width<uintmax_t>(), 0);
  return std::make_unique<LiteralNode>(
      loc, GetBasicType(m_ctx_scope, lldb::eBasicTypeNullPtr), raw_value,
      /*is_literal_zero*/ false);
}

ExprResult DILParser::ParseNumericConstant(clang::Token token) {
  CompilerType bad_type;
  // Parse numeric constant, it can be either integer or float.
  std::string tok_spelling = m_pp->getSpelling(token);

  clang::NumericLiteralParser literal(
      tok_spelling, token.getLocation(), m_pp->getSourceManager(),
      m_pp->getLangOpts(), m_pp->getTargetInfo(), m_pp->getDiagnostics());

  if (literal.hadError) {
    BailOut(
        ErrorCode::kInvalidNumericLiteral,
        "Failed to parse token as numeric-constant: " + TokenDescription(token),
        token.getLocation());
    return std::make_unique<DILErrorNode>(bad_type);
  }

  // Check for floating-literal and integer-literal. Fail on anything else (i.e.
  // fixed-point literal, who needs them anyway??).
  if (literal.isFloatingLiteral()) {
    return ParseFloatingLiteral(literal, token);
  }
  if (literal.isIntegerLiteral()) {
    return ParseIntegerLiteral(literal, token);
  }

  // Don't care about anything else.
  BailOut(ErrorCode::kInvalidNumericLiteral,
          "numeric-constant should be either float or integer literal: " +
              TokenDescription(token),
          token.getLocation());
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::ParseFloatingLiteral(
    clang::NumericLiteralParser& literal,
    clang::Token token) {
  const llvm::fltSemantics& format = literal.isFloat
                                         ? llvm::APFloat::IEEEsingle()
                                         : llvm::APFloat::IEEEdouble();
  llvm::APFloat raw_value(format);
  llvm::APFloat::opStatus result = literal.GetFloatValue(raw_value);

  // Overflow is always an error, but underflow is only an error if we
  // underflowed to zero (APFloat reports denormals as underflow).
  if ((result & llvm::APFloat::opOverflow) ||
      ((result & llvm::APFloat::opUnderflow) && raw_value.isZero())) {
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("float underflow/overflow happened: {0}",
                          TokenDescription(token)),
            token.getLocation());
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto basic_type =
      literal.isFloat ? lldb::eBasicTypeFloat : lldb::eBasicTypeDouble;
  return std::make_unique<LiteralNode>(
      token.getLocation(), GetBasicType(m_ctx_scope, basic_type), raw_value,
      /*is_literal_zero*/ false);
}

ExprResult DILParser::ParseIntegerLiteral(clang::NumericLiteralParser& literal,
                                          clang::Token token) {
  // Create a value big enough to fit all valid numbers.
  llvm::APInt raw_value(type_width<uintmax_t>(), 0);

  if (literal.GetIntegerValue(raw_value)) {
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("integer literal is too large to be represented in "
                          "any integer type: {0}",
                          TokenDescription(token)),
            token.getLocation());
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto [type, is_unsigned] = PickIntegerType(m_ctx_scope, literal, raw_value);

#if LLVM_VERSION_MAJOR < 14
  bool is_literal_zero = raw_value.isNullValue();
#else
  bool is_literal_zero = raw_value.isZero();
#endif

  return std::make_unique<LiteralNode>(token.getLocation(),
                                       GetBasicType(m_ctx_scope, type),
                                       raw_value,
                                       is_literal_zero);
}

// Parse nested_name_specifier.
//
//  nested_name_specifier:
//    type_name "::"
//    namespace_name '::'
//    nested_name_specifier identifier "::"
//    nested_name_specifier simple_template_id "::"
//
std::string DILParser::ParseNestedNameSpecifier() {
  // The first token in nested_name_specifier is always an identifier, or
  // '(anonymous namespace)'.
  if (m_token.isNot(clang::tok::identifier) &&
      m_token.isNot(clang::tok::l_paren)) {
    return "";
  }

  // Anonymous namespaces need to be treated specially: They are represented
  // the the string '(anonymous namespace)', which has a space in it (throwing
  // of normal parsing) and is not actually proper C++> Check to see if we're
  // looking at '(anonymous namespace)::...'
  if (m_token.is(clang::tok::l_paren)) {
    // Look for all the pieces, in order:
    // l_paren 'anonymous' 'namespace' r_paren coloncolon
    if (m_pp->LookAhead(0).is(clang::tok::identifier)
        && (m_pp->getSpelling(m_pp->LookAhead(0)) == "anonymous")
        && m_pp->LookAhead(1).is(clang::tok::kw_namespace)
        && m_pp->LookAhead(2).is(clang::tok::r_paren)
        && m_pp->LookAhead(3).is(clang::tok::coloncolon)) {
      ConsumeToken(); // l_paren
      ConsumeToken(); // identifier 'anonymous'
      ConsumeToken(); // keyword 'namespace'
      ConsumeToken(); // r_paren
      ConsumeToken(); // coloncolon

      assert ((m_token.is(clang::tok::identifier)
               || m_token.is(clang::tok::l_paren)) &&
              "Expected an identifier or anonymous namespace, but not found.");
      // Continue parsing the nested_namespace_specifier.
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        Expect(clang::tok::identifier);
        identifier2 = m_pp->getSpelling(m_token);
        ConsumeToken();
      }
      return "(anonymous namespace)::" + identifier2;
    } else {
      return "";
    }
  } // end of special handling for '(anonymous namespace)'

  // If the next token is scope ("::"), then this is indeed a
  // nested_name_specifier
  if (m_pp->LookAhead(0).is(clang::tok::coloncolon)) {
    // This nested_name_specifier is a single identifier.
    std::string identifier = m_pp->getSpelling(m_token);
    ConsumeToken();
    Expect(clang::tok::coloncolon);
    ConsumeToken();
    // Continue parsing the nested_name_specifier.
    return identifier + "::" + ParseNestedNameSpecifier();
  }

  // If the next token starts a template argument list, then we have a
  // simple_template_id here.
  if (m_pp->LookAhead(0).is(clang::tok::less)) {
    // We don't know whether this will be a nested_name_identifier or just a
    // type_name. Prepare to rollback if this is not a nested_name_identifier.
    TentativeParsingAction tentative_parsing(this);

    // TODO(werat): Parse just the simple_template_id?
    auto type_name = ParseTypeName();

    // If we did parse the type_name successfully and it's followed by the scope
    // operator ("::"), then this is indeed a nested_name_specifier. Commit the
    // tentative parsing and continue parsing nested_name_specifier.
    if (!type_name.empty() && m_token.is(clang::tok::coloncolon)) {
      tentative_parsing.Commit();
      ConsumeToken();
      // Continue parsing the nested_name_specifier.
      return type_name + "::" + ParseNestedNameSpecifier();
    }

    // Not a nested_name_specifier, but could be just a type_name or something
    // else entirely. Rollback the parser and try a different path.
    tentative_parsing.Rollback();
  }

  return "";
}

// Parse a type_name.
//
//  type_name:
//    class_name
//    enum_name
//    typedef_name
//    simple_template_id
//
//  class_name
//    identifier
//
//  enum_name
//    identifier
//
//  typedef_name
//    identifier
//
//  simple_template_id:
//    template_name "<" [template_argument_list] ">"
//
std::string DILParser::ParseTypeName() {
  // Typename always starts with an identifier.
  if (m_token.isNot(clang::tok::identifier)) {
    return "";
  }

  // If the next token starts a template argument list, parse this type_name as
  // a simple_template_id.
  if (m_pp->LookAhead(0).is(clang::tok::less)) {
    // Parse the template_name. In this case it's just an identifier.
    std::string template_name = m_pp->getSpelling(m_token);
    ConsumeToken();
    // Consume the "<" token.
    ConsumeToken();

    // Short-circuit for missing template_argument_list.
    if (m_token.is(clang::tok::greater)) {
      ConsumeToken();
      return llvm::formatv("{0}<>", template_name);
    }

    // Try parsing template_argument_list.
    auto template_argument_list = ParseTemplateArgumentList();

    if (m_token.is(clang::tok::greater)) {
      // Single closing angle bracket is a valid end of the template argument
      // list, just consume it.
      ConsumeToken();

    } else if (m_token.is(clang::tok::greatergreater)) {
      // C++11 allows using ">>" in nested template argument lists and C++-style
      // casts. In this case we alter change the token type to ">", but don't
      // consume it -- it will be done on the outer level when completing the
      // outer template argument list or C++-style cast.
      m_token.setKind(clang::tok::greater);

    } else {
      // Not a valid end of the template argument list, failed to parse a
      // simple_template_id
      return "";
    }

    return llvm::formatv("{0}<{1}>", template_name, template_argument_list);
  }

  // Otherwise look for a class_name, enum_name or a typedef_name.
  std::string identifier = m_pp->getSpelling(m_token);
  ConsumeToken();

  return identifier;
}

// Parse a template_argument_list.
//
//  template_argument_list:
//    template_argument
//    template_argument_list "," template_argument
//
std::string DILParser::ParseTemplateArgumentList() {
  // Parse template arguments one by one.
  std::vector<std::string> arguments;

  do {
    // Eat the comma if this is not the first iteration.
    if (arguments.size() > 0) {
      ConsumeToken();
    }

    // Try parsing a template_argument. If this fails, then this is actually not
    // a template_argument_list.
    auto argument = ParseTemplateArgument();
    if (argument.empty()) {
      return "";
    }

    arguments.push_back(argument);

  } while (m_token.is(clang::tok::comma));

  // Internally in LLDB/Clang nested template type names have extra spaces to
  // avoid having ">>". Add the extra space before the closing ">" if the
  // template argument is also a template.
  if (arguments.back().back() == '>') {
    arguments.back().push_back(' ');
  }

  return llvm::formatv("{0:$[, ]}",
                       llvm::make_range(arguments.begin(), arguments.end()));
}

// Parse a template_argument.
//
//  template_argument:
//    type_id
//    numeric_literal
//    id_expression
//
std::string DILParser::ParseTemplateArgument() {
  // There is no way to know at this point whether there is going to be a
  // type_id or something else. Try different options one by one.

  {
    // [temp.arg](http://eel.is/c++draft/temp.arg#2)
    //
    // In a template-argument, an ambiguity between a type-id and an expression
    // is resolved to a type-id, regardless of the form of the corresponding
    // template-parameter.

    // Therefore, first try parsing type_id.
    TentativeParsingAction tentative_parsing(this);

    auto type_id = ParseTypeId();
    if (type_id) {
      tentative_parsing.Commit();

      CompilerType type = type_id.value();
      return type.IsValid()
          ? std::string(type.GetTypeName().AsCString())
          : "";

    } else {
      // Failed to parse a type_id. Rollback the parser and try something else.
      tentative_parsing.Rollback();
    }
  }

  {
    // The next candidate is a numeric_literal.
    TentativeParsingAction tentative_parsing(this);

    // Parse a numeric_literal.
    if (m_token.is(clang::tok::numeric_constant)) {
      // TODO(werat): Actually parse the literal, check if it's valid and
      // canonize it (e.g. 8LL -> 8).
      std::string numeric_literal = m_pp->getSpelling(m_token);
      ConsumeToken();

      if (TokenEndsTemplateArgumentList(m_token)) {
        tentative_parsing.Commit();
        return numeric_literal;
      }
    }

    // Failed to parse a numeric_literal.
    tentative_parsing.Rollback();
  }

  {
    // The next candidate is an id_expression.
    TentativeParsingAction tentative_parsing(this);

    // Parse an id_expression.
    auto id_expression = ParseIdExpression();

    // If we've parsed the id_expression successfully and the next token can
    // finish the template_argument, then we're done here.
    if (!id_expression.empty() && TokenEndsTemplateArgumentList(m_token)) {
      tentative_parsing.Commit();
      return id_expression;
    }
    // Failed to parse a id_expression.
    tentative_parsing.Rollback();
  }

  // TODO(b/164399865): Another valid option here is a constant_expression, but
  // we definitely don't want to support constant arithmetic like "Foo<1+2>".
  // We can probably use ParsePrimaryExpression here, but need to figure out the
  // "stringification", since ParsePrimaryExpression returns ExprResult (and
  // potentially a whole expression, not just a single constant.)

  // This is not a template_argument.
  return "";
}

// Parse a ptr_operator.
//
//  ptr_operator:
//    "*" [cv_qualifier_seq]
//    "&"
//
DILParser::PtrOperator DILParser::ParsePtrOperator() {
  ExpectOneOf(clang::tok::star, clang::tok::amp);

  PtrOperator ptr_operator;
  if (m_token.is(clang::tok::star)) {
    ptr_operator = std::make_tuple(clang::tok::star, m_token.getLocation());
    ConsumeToken();

    //
    //  cv_qualifier_seq:
    //    cv_qualifier [cv_qualifier_seq]
    //
    //  cv_qualifier:
    //    "const"
    //    "volatile"
    //
    while (IsCvQualifier(m_token)) {
      // Just ignore CV quialifiers, we don't use them in type casting.
      ConsumeToken();
    }

  } else if (m_token.is(clang::tok::amp)) {
    ptr_operator = std::make_tuple(clang::tok::amp, m_token.getLocation());
    ConsumeToken();
  }

  return ptr_operator;
}

// Parse a builtin_func.
//
//  builtin_func:
//    builtin_func_name "(" [builtin_func_argument_list] ")"
//
//  builtin_func_name:
//    "__log2"
//
//  builtin_func_argument_list:
//    builtin_func_argument
//    builtin_func_argument_list "," builtin_func_argument
//
//  builtin_func_argument:
//    expression
//
ExprResult DILParser::ParseBuiltinFunction(
    clang::SourceLocation loc, std::unique_ptr<BuiltinFunctionDef> func_def) {
  Expect(clang::tok::l_paren);
  ConsumeToken();

  std::vector<ExprResult> arguments;
  CompilerType bad_type;

  if (m_token.is(clang::tok::r_paren)) {
    // Empty argument list, nothing to do here.
    ConsumeToken();
  } else {
    // Non-empty argument list, parse all the arguments.
    do {
      // Eat the comma if this is not the first iteration.
      if (arguments.size() > 0) {
        ConsumeToken();
      }

      // Parse a builtin_func_argument. If failed to parse, bail out early and
      // don't try parsing the rest of the arguments.
      auto argument = ParseExpression();
      if (argument->is_error()) {
        return std::make_unique<DILErrorNode>(bad_type);
      }

      arguments.push_back(std::move(argument));

    } while (m_token.is(clang::tok::comma));

    Expect(clang::tok::r_paren);
    ConsumeToken();
  }

  // Check we have the correct number of arguments.
  if (arguments.size() != func_def->m_arguments.size()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(
                "no matching function for call to '{0}': requires {1} "
                "argument(s), but {2} argument(s) were provided",
                func_def->m_name, func_def->m_arguments.size(),
                arguments.size()),
            loc);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  // Now check that all arguments are correct types and perform implicit
  // conversions if possible.
  for (size_t i = 0; i < arguments.size(); ++i) {
    // HACK: Void means "any" and we'll check in runtime. The argument will be
    // passed as is without any conversions.
    if (func_def->m_arguments[i].GetBasicTypeEnumeration()
        == lldb::eBasicTypeVoid) {
      continue;
    }
    arguments[i] = InsertImplicitConversion(std::move(arguments[i]),
                                            func_def->m_arguments[i]);
    if (arguments[i]->is_error()) {
      return std::make_unique<DILErrorNode>(bad_type);
    }
  }

  return std::make_unique<BuiltinFunctionCallNode>(
      loc, func_def->m_return_type, func_def->m_name, std::move(arguments));
}

ExprResult DILParser::BuildCStyleCast(CompilerType type, ExprResult rhs,
                                      clang::SourceLocation location) {
  CStyleCastKind kind;
  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();

  // Cast to basic type (integer/float).
  if (IsScalar(type)) {
    // Before casting arrays to scalar types, array-to-pointer conversion
    // should be performed.
    if (IsArrayType(rhs_type)) {
      rhs = InsertArrayToPointerConversion(std::move(rhs));
      rhs_type = rhs->result_type_deref();
    }
    // Pointers can be cast to integers of the same or larger size.
    if (IsPointerType(rhs_type) || IsNullPtrType(rhs_type)) {
      // C-style cast from pointer to float/double is not allowed.
      if (IsFloat(type)) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("C-style cast from {0} to {1} is not allowed",
                              TypeDescription(rhs_type), TypeDescription(type)),
                location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      // Casting pointer to bool is valid. Otherwise check if the result type
      // is at least as big as the pointer size.
      uint64_t type_byte_size = 0;
      uint64_t rhs_type_byte_size = 0;
      if (auto temp = type.GetByteSize(m_ctx_scope.get()))
        type_byte_size = temp.value();
      if (auto temp = rhs_type.GetByteSize(m_ctx_scope.get()))
        rhs_type_byte_size = temp.value();
      if (!IsBool(type) && type_byte_size < rhs_type_byte_size) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cast from pointer to smaller type {0} loses information",
                    TypeDescription(type)),
                location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
    } else if (!IsScalar(rhs_type) && !IsEnum(rhs_type)) {
      // Otherwise accept only arithmetic types and enums.
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot convert {0} to {1} without a conversion operator",
                  TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    kind = CStyleCastKind::kArithmetic;

  } else if (IsEnum(type)) {
    // Cast to enum type.
    if (!IsScalar(rhs_type) && !IsEnum(rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    kind = CStyleCastKind::kEnumeration;

  } else if (IsPointerType(type)) {
    // Cast to pointer type.
    if (!IsInteger(rhs_type) && !IsEnum(rhs_type) &&
        !IsArrayType(rhs_type) && !IsPointerType(rhs_type) &&
        !IsNullPtrType(rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("cannot cast from type {0} to pointer type {1}",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    kind = CStyleCastKind::kPointer;

  } else if (IsNullPtrType(type)) {
    // Cast to nullptr type.
    if (!IsNullPtrType(type) && !rhs->is_literal_zero()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    kind = CStyleCastKind::kNullptr;

  } else if (IsReferenceType(type)) {
    // Cast to a reference type.
    if (rhs->is_rvalue()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from rvalue to reference type {0}",
                            TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    kind = CStyleCastKind::kReference;

  } else {
    // Unsupported cast.
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("casting of {0} to {1} is not implemented yet",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CStyleCastNode>(location, type, std::move(rhs), kind);
}

ExprResult DILParser::BuildCxxCast(clang::tok::TokenKind kind, CompilerType type,
                                   ExprResult rhs,
                                   clang::SourceLocation location) {
  assert((kind == clang::tok::kw_static_cast ||
          kind == clang::tok::kw_dynamic_cast ||
          kind == clang::tok::kw_reinterpret_cast) &&
         "invalid C++-style cast type");

  // TODO(werat): Implement custom builders for all C++-style casts.
  if (kind == clang::tok::kw_dynamic_cast) {
    return BuildCxxDynamicCast(type, std::move(rhs), location);
  }
  if (kind == clang::tok::kw_reinterpret_cast) {
    return BuildCxxReinterpretCast(type, std::move(rhs), location);
  }
  if (kind == clang::tok::kw_static_cast) {
    return BuildCxxStaticCast(type, std::move(rhs), location);
  }
  return BuildCStyleCast(type, std::move(rhs), location);
}

ExprResult DILParser::BuildCxxStaticCast(CompilerType type, ExprResult rhs,
                                         clang::SourceLocation location) {
  auto rhs_type = rhs->result_type_deref();

  // Perform implicit array-to-pointer conversion.
  if (IsArrayType(rhs_type)) {
    rhs = InsertArrayToPointerConversion(std::move(rhs));
    rhs_type = rhs->result_type_deref();
  }

  if (CompareTypes(rhs_type, type)) {
    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               CxxStaticCastKind::kNoOp,
                                               /*is_rvalue*/ true);
  }

  if (IsScalar(type)) {
    return BuildCxxStaticCastToScalar(type, std::move(rhs), location);
  } else if (IsEnum(type)) {
    return BuildCxxStaticCastToEnum(type, std::move(rhs), location);
  } else if (IsPointerType(type)) {
    return BuildCxxStaticCastToPointer(type, std::move(rhs), location);
  } else if (IsNullPtrType(type)) {
    return BuildCxxStaticCastToNullPtr(type, std::move(rhs), location);
  } else if (IsReferenceType(type)) {
    return BuildCxxStaticCastToReference(type, std::move(rhs), location);
  }

  // Unsupported cast.
  BailOut(ErrorCode::kNotImplemented,
          llvm::formatv("casting of {0} to {1} is not implemented yet",
                        TypeDescription(rhs_type), TypeDescription(type)),
          location);
  CompilerType bad_type;
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildCxxStaticCastToScalar(CompilerType type,
                                                 ExprResult rhs,
                                                 clang::SourceLocation location)
{
  auto rhs_type = rhs->result_type_deref();
  CompilerType bad_type;

  if (IsPointerType(rhs_type) || IsNullPtrType(rhs_type)) {
    // Pointers can be casted to bools.
    if (!IsBool(type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("static_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
  } else if (!IsScalar(rhs_type) && !IsEnum(rhs_type)) {
    // Otherwise accept only arithmetic types and enums.
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("cannot convert {0} to {1} without a conversion operator",
                      TypeDescription(rhs_type), TypeDescription(type)),
        location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::kArithmetic,
                                             /*is_rvalue*/ true);
}

ExprResult DILParser::BuildCxxStaticCastToEnum(CompilerType type, ExprResult rhs,
                                               clang::SourceLocation location) {
  auto rhs_type = rhs->result_type_deref();

  if (!IsScalar(rhs_type) && !IsEnum(rhs_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("static_cast from {0} to {1} is not allowed",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::kEnumeration,
                                             /*is_rvalue*/ true);
}

ExprResult DILParser::BuildCxxStaticCastToPointer(CompilerType type,
                                                  ExprResult rhs,
                                                  clang::SourceLocation location)
{
  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();

  if (IsPointerType(rhs_type)) {
    auto type_pointee = GetPointeeType(type);
    auto rhs_type_pointee = GetPointeeType(rhs_type);

    if (IsRecordType(type_pointee) && IsRecordType(rhs_type_pointee)) {
      return BuildCxxStaticCastForInheritedTypes(type, std::move(rhs),
                                                 location);
    }

    if (!IsPointerToVoid(type) && !IsPointerToVoid(rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("static_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
  } else if (!IsNullPtrType(rhs_type) && !rhs->is_literal_zero()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot cast from type {0} to pointer type '{1}'",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::kPointer,
                                             /*is_rvalue*/ true);
}

ExprResult DILParser::BuildCxxStaticCastToNullPtr(CompilerType type,
                                                  ExprResult rhs,
                                                  clang::SourceLocation location)
{
  auto rhs_type = rhs->result_type_deref();

  if (!IsNullPtrType(rhs_type) && !rhs->is_literal_zero()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("static_cast from {0} to {1} is not allowed",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    CompilerType bad_type;
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::kNullptr,
                                             /*is_rvalue*/ true);
}

ExprResult DILParser::BuildCxxStaticCastToReference(
    CompilerType type,
    ExprResult rhs,
    clang::SourceLocation location) {
  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();
  auto type_deref = GetDereferencedType(type);

  if (rhs->is_rvalue()) {
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("static_cast from rvalue of type {0} to reference "
                          "type {1} is not implemented yet",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  if (CompareTypes(type_deref, rhs_type)) {
    return std::make_unique<CxxStaticCastNode>(
        location, type_deref, std::move(rhs), CxxStaticCastKind::kNoOp,
        /*is_rvalue*/ false);
  }

  if (IsRecordType(type_deref) && IsRecordType(rhs_type)) {
    return BuildCxxStaticCastForInheritedTypes(type, std::move(rhs), location);
  }

  BailOut(ErrorCode::kNotImplemented,
          llvm::formatv("static_cast from {0} to {1} is not implemented yet",
                        TypeDescription(rhs_type), TypeDescription(type)),
          location);
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildCxxStaticCastForInheritedTypes(
    CompilerType type, ExprResult rhs, clang::SourceLocation location) {
  assert((IsPointerType(type) || IsReferenceType(type)) &&
         "target type should either be a pointer or a reference");

  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();
  auto record_type = IsPointerType(type)
                     ? GetPointeeType(type)
                     : GetDereferencedType(type);
  auto rhs_record_type =
      IsPointerType(rhs_type) ? GetPointeeType(rhs_type) : rhs_type;

  assert(IsRecordType(record_type) && IsRecordType(rhs_record_type) &&
         "underlying RHS and target types should be record types");
  assert(!CompareTypes(record_type, rhs_record_type) &&
         "underlying RHS and target types should be different");

  // Result of cast to reference type is an lvalue.
  bool is_rvalue = !IsReferenceType(type);

  // Handle derived-to-base conversion.
  std::vector<uint32_t> idx;
  if (GetPathToBaseType(rhs_record_type, record_type, &idx,
                        /*offset*/ nullptr)) {
    std::reverse(idx.begin(), idx.end());
    // At this point `idx` represents indices of direct base classes on path
    // from the `rhs` type to the target `type`.
    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               std::move(idx), is_rvalue);
  }

  // Handle base-to-derived conversion.
  uint64_t offset = 0;
  if (GetPathToBaseType(record_type, rhs_record_type, /*path*/ nullptr,
                        &offset)) {
    CompilerType virtual_base;
    if (IsVirtualBase(record_type, rhs_record_type, &virtual_base)) {
      // Base-to-derived conversion isn't possible for virtually inherited
      // types (either directly or indirectly).
      assert(virtual_base.IsValid() && "virtual base should be valid");
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("cannot cast {0} to {1} via virtual base {2}",
                            TypeDescription(rhs_type), TypeDescription(type),
                            TypeDescription(virtual_base)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }

    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               offset, is_rvalue);
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("static_cast from {0} to {1}, which are not "
                        "related by inheritance, is not allowed",
                        TypeDescription(rhs_type), TypeDescription(type)),
          location);
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildCxxReinterpretCast(CompilerType type, ExprResult rhs,
                                              clang::SourceLocation location) {
  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();
  bool is_rvalue = true;

  if (IsScalar(type)) {
    // reinterpret_cast doesn't support non-integral scalar types.
    if (!IsInteger(type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }

    // Perform implicit conversions.
    if (IsArrayType(rhs_type)) {
      rhs = InsertArrayToPointerConversion(std::move(rhs));
      rhs_type = rhs->result_type_deref();
    }

    if (IsPointerType(rhs_type) || IsNullPtrType(rhs_type)) {
      // A pointer can be converted to any integral type large enough to hold
      // its value.
      uint64_t type_byte_size = 0;
      uint64_t rhs_type_byte_size = 0;
      if (auto temp = type.GetByteSize(m_ctx_scope.get()))
        type_byte_size = temp.value();
      if (auto temp = rhs_type.GetByteSize(m_ctx_scope.get()))
        rhs_type_byte_size = temp.value();
      if (type_byte_size < rhs_type_byte_size) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cast from pointer to smaller type {0} loses information",
                    TypeDescription(type)),
                location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
    } else if (!CompareTypes(type, rhs_type)) {
      // Integral type can be converted to its own type.
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
  } else if (IsEnum(type)) {
    // Enumeration type can be converted to its own type.
    if (!CompareTypes(type, rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }

  } else if (IsPointerType(type)) {
    // Integral, enumeration and other pointer types can be converted to any
    // pointer type.
    // TODO: Implement an explicit node for array-to-pointer conversions.
    if (!IsInteger(rhs_type) && !IsEnum(rhs_type) &&
        !IsArrayType(rhs_type) && !IsPointerType(rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            TypeDescription(rhs_type), TypeDescription(type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }

  } else if (IsNullPtrType(type)) {
    // reinterpret_cast to nullptr_t isn't allowed (even for nullptr_t).
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);

  } else if (IsReferenceType(type)) {
    // L-values can be converted to any reference type.
    if (rhs->is_rvalue()) {
      BailOut(
          ErrorCode::kInvalidOperandType,
          llvm::formatv("reinterpret_cast from rvalue to reference type {0}",
                        TypeDescription(type)),
          location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
    // Casting to reference types gives an L-value result.
    is_rvalue = false;

  } else {
    // Unsupported cast.
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("casting of {0} to {1} is not implemented yet",
                          TypeDescription(rhs_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<CxxReinterpretCastNode>(location, type,
                                                  std::move(rhs), is_rvalue);
}

ExprResult DILParser::BuildCxxDynamicCast(CompilerType type, ExprResult rhs,
                                          clang::SourceLocation location) {
  CompilerType pointee_type;
  CompilerType bad_type;
  if (IsPointerType(type)) {
    pointee_type = GetPointeeType(type);
  } else if (IsReferenceType(type)) {
    pointee_type = GetDereferencedType(type);
  } else {
    // Dynamic casts are allowed only for pointers and references.
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("invalid target type {0} for dynamic_cast; target type "
                      "must be a reference or pointer type to a defined class",
                      TypeDescription(type)),
        location);
    return std::make_unique<DILErrorNode>(bad_type);
  }
  // Dynamic casts are allowed only for record types.
  if (!IsRecordType(pointee_type)) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("{0} is not a class type", TypeDescription(pointee_type)),
        location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto expr_type = rhs->result_type();
  if (IsPointerType(expr_type)) {
    expr_type = GetPointeeType(expr_type);
  } else if (IsReferenceType(expr_type)) {
    expr_type = GetDereferencedType(expr_type);
  } else {
    // Expression type must be a pointer or a reference.
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot use dynamic_cast to convert from {0} to {1}",
                          TypeDescription(expr_type), TypeDescription(type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }
  // Dynamic casts are allowed only for record types.
  if (!IsRecordType(expr_type)) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("{0} is not a class type", TypeDescription(expr_type)),
        location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  // Expr type must be polymorphic.
  if (!IsPolymorphicClass(expr_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("{0} is not polymorphic", TypeDescription(expr_type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  // LLDB doesn't support dynamic_cast in the expression evaluator. We disable
  // it too to match the behaviour, but theoretically it can be implemented.
  BailOut(ErrorCode::kInvalidOperandType,
          "dynamic_cast is not supported in this context", location);
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildUnaryOp(UnaryOpKind kind, ExprResult rhs,
                                clang::SourceLocation location) {
  CompilerType result_type;
  auto rhs_type = rhs->result_type_deref();
  CompilerType bad_type;

  switch (kind) {
    case UnaryOpKind::Deref: {
      if (IsPointerType(rhs_type)) {
        result_type = GetPointeeType(rhs_type);
      } else if (IsSmartPtrType(rhs_type)) {
        rhs = InsertSmartPtrToPointerConversion(std::move(rhs));
        result_type = GetPointeeType(rhs->result_type_deref());
      } else if (IsArrayType(rhs_type)) {
        rhs = InsertArrayToPointerConversion(std::move(rhs));
        result_type = GetPointeeType(rhs->result_type_deref());
      } else {
        BailOut(
            ErrorCode::kInvalidOperandType,
            llvm::formatv("indirection requires pointer operand ({0} invalid)",
                          TypeDescription(rhs_type)),
            location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      break;
    }
    case UnaryOpKind::AddrOf: {
      if (rhs->is_rvalue()) {
        BailOut(
            ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot take the address of an rvalue of type {0}",
                          TypeDescription(rhs_type)),
            location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      if (rhs->is_bitfield()) {
        BailOut(ErrorCode::kInvalidOperandType,
                "address of bit-field requested", location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
      result_type = GetPointerType(rhs_type);
      break;
    }
    case UnaryOpKind::Plus:
    case UnaryOpKind::Minus: {
      rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));
      rhs_type = rhs->result_type_deref();
      if (IsScalar(rhs_type) ||
          // Unary plus is allowed for pointers.
          (kind == UnaryOpKind::Plus && IsPointerType(rhs_type))) {
        result_type = rhs->result_type();
      }
      break;
    }
    case UnaryOpKind::Not: {
      rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));
      rhs_type = rhs->result_type_deref();
      if (IsInteger(rhs_type)) {
        result_type = rhs->result_type();
      }
      break;
    }
    case UnaryOpKind::LNot: {
      if (IsContextuallyConvertibleToBool(rhs_type)) {
        result_type = GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);
      }
      break;
    }
    case UnaryOpKind::PostInc:
    case UnaryOpKind::PostDec:
    case UnaryOpKind::PreInc:
    case UnaryOpKind::PreDec: {
      return BuildIncrementDecrement(kind, std::move(rhs), location);
    }

    default:
      llvm_unreachable("invalid unary op kind");
  }

  if (!result_type) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(kInvalidOperandsToUnaryExpression,
                          TypeDescription(rhs_type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<UnaryOpNode>(location, result_type, kind,
                                       std::move(rhs));
}

ExprResult DILParser::BuildBinaryOp(BinaryOpKind kind, ExprResult lhs,
                                 ExprResult rhs,
                                 clang::SourceLocation location) {
  // TODO(werat): Get the "original" type (i.e. the one before implicit casts)
  // from the ExprResult.
  auto orig_lhs_type = lhs->result_type_deref();
  auto orig_rhs_type = rhs->result_type_deref();

  // Result type of the binary expression. For example, for `char + int` the
  // result type is `int`, but for `char += int` the result type is `char`.
  CompilerType result_type;

  // In case binary operation is a composite assignment, the type of the binary
  // expression _before_ the assignment. For example, for `char += int`
  // composite assignment type is `int`, because `char + int` is promoted to
  // `int + int`.
  CompilerType comp_assign_type;

  switch (kind) {
    case BinaryOpKind::Add:
      result_type =
          PrepareBinaryAddition(lhs, rhs, location, /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Sub:
      result_type = PrepareBinarySubtraction(lhs, rhs, location,
                                             /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Mul:
    case BinaryOpKind::Div:
      result_type = PrepareBinaryMulDiv(lhs, rhs,
                                        /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Rem:
      result_type = PrepareBinaryRemainder(lhs, rhs, /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::And:
    case BinaryOpKind::Or:
    case BinaryOpKind::Xor:
      result_type = PrepareBinaryBitwise(lhs, rhs,
                                         /*is_comp_assign*/ false);
      break;
    case BinaryOpKind::Shl:
    case BinaryOpKind::Shr:
      result_type = PrepareBinaryShift(lhs, rhs,
                                       /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::EQ:
    case BinaryOpKind::NE:
    case BinaryOpKind::LT:
    case BinaryOpKind::LE:
    case BinaryOpKind::GT:
    case BinaryOpKind::GE:
      result_type = PrepareBinaryComparison(kind, lhs, rhs, location);
      break;

    case BinaryOpKind::LAnd:
    case BinaryOpKind::LOr:
      result_type = PrepareBinaryLogical(lhs, rhs);
      break;

    case BinaryOpKind::Assign:
      // For plain assignment try to implicitly convert RHS to the type of LHS.
      // Later we'll check if the assignment is actually possible.
      rhs = InsertImplicitConversion(std::move(rhs), lhs->result_type_deref());
      // Shortcut for the case when the implicit conversion is not possible.
      if (rhs->is_error()) {
        CompilerType bad_type;
        return std::make_unique<DILErrorNode>(bad_type);
      }
      comp_assign_type = rhs->result_type_deref();
      break;

    case BinaryOpKind::AddAssign:
      comp_assign_type = PrepareBinaryAddition(lhs, rhs, location,
                                               /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::SubAssign:
      comp_assign_type = PrepareBinarySubtraction(lhs, rhs, location,
                                                  /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::MulAssign:
    case BinaryOpKind::DivAssign:
      comp_assign_type = PrepareBinaryMulDiv(lhs, rhs,
                                             /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::RemAssign:
      comp_assign_type =
          PrepareBinaryRemainder(lhs, rhs, /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::AndAssign:
    case BinaryOpKind::OrAssign:
    case BinaryOpKind::XorAssign:
      comp_assign_type = PrepareBinaryBitwise(lhs, rhs,
                                              /*is_comp_assign*/ true);
      break;
    case BinaryOpKind::ShlAssign:
    case BinaryOpKind::ShrAssign:
      comp_assign_type = PrepareBinaryShift(lhs, rhs,
                                            /*is_comp_assign*/ true);
      break;

    default:
      llvm_unreachable("invalid binary op kind");
  }

  // If we're building a composite assignment, check for composite assignments
  // constraints: if the LHS is assignable, if the type of the binary operation
  // can be assigned to it, etc.
  if (comp_assign_type.IsValid()) {
    result_type = PrepareCompositeAssignment(comp_assign_type, lhs, location);
  }

  // If the result type is valid, then the binary operation is valid!
  if (result_type.IsValid()) {
    return std::make_unique<BinaryOpNode>(location, result_type, kind,
                                          std::move(lhs), std::move(rhs),
                                          comp_assign_type);
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv(kInvalidOperandsToBinaryExpression,
                        TypeDescription(orig_lhs_type),
                        TypeDescription(orig_rhs_type)),
          location);
  CompilerType bad_type;
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildTernaryOp(ExprResult cond, ExprResult lhs,
                                  ExprResult rhs,
                                  clang::SourceLocation location) {
  CompilerType bad_type;
  // First check if the condition contextually converted to bool.
  auto cond_type = cond->result_type_deref();
  if (!IsContextuallyConvertibleToBool(cond_type)) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, TypeDescription(cond_type)),
        location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  // If operands have the same type, don't do any promotions.
  if (CompareTypes(lhs_type, rhs_type)) {
    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If both operands have arithmetic type, apply the usual arithmetic
  // conversions to bring them to a common type.
  if (IsScalarOrUnscopedEnum(lhs_type) &&
      IsScalarOrUnscopedEnum(rhs_type)) {
    auto result_type = UsualArithmeticConversions(m_ctx_scope, lhs, rhs);
    return std::make_unique<TernaryOpNode>(
        location, result_type, std::move(cond), std::move(lhs), std::move(rhs));
  }

  // Apply array-to-pointer implicit conversions.
  if (IsArrayType(lhs_type)) {
    lhs = InsertArrayToPointerConversion(std::move(lhs));
    lhs_type = lhs->result_type_deref();
  }
  if (IsArrayType(rhs_type)) {
    rhs = InsertArrayToPointerConversion(std::move(rhs));
    rhs_type = rhs->result_type_deref();
  }

  // Check if operands have the same pointer type.
  if (CompareTypes(lhs_type, rhs_type)) {
    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If one operand is a pointer and the other is a nullptr or literal zero,
  // convert the nullptr operand to pointer type.
  if (IsPointerType(lhs_type) &&
      (rhs->is_literal_zero() || IsNullPtrType(rhs_type))) {
    rhs = std::make_unique<CStyleCastNode>(
        rhs->location(), lhs_type, std::move(rhs), CStyleCastKind::kPointer);

    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }
  if ((lhs->is_literal_zero() || IsNullPtrType(lhs_type)) &&
      IsPointerType(rhs_type)) {
    lhs = std::make_unique<CStyleCastNode>(
        lhs->location(), rhs_type, std::move(lhs), CStyleCastKind::kPointer);

    return std::make_unique<TernaryOpNode>(location, rhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If one operand is nullptr and the other one is literal zero, convert
  // the literal zero to a nullptr type.
  if (IsNullPtrType(lhs_type) && rhs->is_literal_zero()) {
    rhs = std::make_unique<CStyleCastNode>(
        rhs->location(), lhs_type, std::move(rhs), CStyleCastKind::kNullptr);

    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }
  if (lhs->is_literal_zero() && IsNullPtrType(rhs_type)) {
    lhs = std::make_unique<CStyleCastNode>(
        lhs->location(), rhs_type, std::move(lhs), CStyleCastKind::kNullptr);

    return std::make_unique<TernaryOpNode>(location, rhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("incompatible operand types ({0} and {1})",
                        TypeDescription(lhs_type), TypeDescription(rhs_type)),
          location);
  return std::make_unique<DILErrorNode>(bad_type);
}

ExprResult DILParser::BuildBinarySubscript(ExprResult lhs, ExprResult rhs,
                                        clang::SourceLocation location) {
  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely
  // equivalent to the expression *((e1)+(e2)).
  // We need to figure out which expression is "base" and which is "index".

  ExprResult base;
  ExprResult index;
  CompilerType bad_type;

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  if (IsArrayType(lhs_type)) {
    base = InsertArrayToPointerConversion(std::move(lhs));
    index = std::move(rhs);
  } else if (IsPointerType(lhs_type)) {
    base = std::move(lhs);
    index = std::move(rhs);
  } else if (IsArrayType(rhs_type)) {
    base = InsertArrayToPointerConversion(std::move(rhs));
    index = std::move(lhs);
  } else if (IsPointerType(rhs_type)) {
    base = std::move(rhs);
    index = std::move(lhs);
  } else {
    // Check to see if this might be a synthetic value.
    const DILAstNode* ast_node = lhs.get();
    if (ast_node->what_am_i() == DILNodeKind::kIdentifierNode) {
      const IdentifierNode* id_node =
          static_cast<const IdentifierNode*>(ast_node);
      auto identifier = static_cast<const IdentifierInfo&>(id_node->info());
      lldb::ValueObjectSP lhs_valobj_sp = identifier.value();
      if (lhs_valobj_sp->HasSyntheticValue()) {
        base = std::move(lhs);
        index = std::move(rhs);
      } else {
        BailOut(ErrorCode::kInvalidOperandType,
                "subscripted value is not an array or pointer", location);
        return std::make_unique<DILErrorNode>(bad_type);
      }
    } else {
      BailOut(ErrorCode::kInvalidOperandType,
              "subscripted value is not an array or pointer", location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
  }

  // Index can be a typedef of a typedef of a typedef of a typedef...
  // Get canonical underlying type.
  auto index_type = index->result_type_deref();

  // Check if the index is of an integral type.
  if (!IsIntegerOrUnscopedEnum(index_type)) {
    BailOut(ErrorCode::kInvalidOperandType, "array subscript is not an integer",
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  auto base_type = base->result_type_deref();
  if (IsPointerToVoid(base_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            "subscript of pointer to incomplete type 'void'", location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<ArraySubscriptNode>(
      location, GetPointeeType(base->result_type_deref()), std::move(base),
      std::move(index));
}

ExprResult DILParser::BuildMemberOf(ExprResult lhs, std::string member_id,
                                    bool is_arrow,
                                    clang::SourceLocation location) {
  CompilerType bad_type;
  auto lhs_type = lhs->result_type_deref();

  if (is_arrow) {
    // "member of pointer" operator, check that LHS is a pointer and
    // dereference it.
    if (!IsPointerType(lhs_type) && !IsSmartPtrType(lhs_type) &&
        !IsArrayType(lhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("member reference type {0} is not a pointer; did "
                            "you mean to use '.'?",
                            TypeDescription(lhs_type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }

    if (IsSmartPtrType(lhs_type)) {
      // If LHS is a smart pointer, decay it to an underlying object.
      lhs = InsertSmartPtrToPointerConversion(std::move(lhs));
      lhs_type = lhs->result_type_deref();
    } else if (IsArrayType(lhs_type)) {
      // If LHS is an array, convert it to pointer.
      lhs = InsertArrayToPointerConversion(std::move(lhs));
      lhs_type = lhs->result_type_deref();
    }

    lhs_type = GetPointeeType(lhs_type);
  } else {
    // "member of object" operator, check that LHS is an object.
    if (IsPointerType(lhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("member reference type {0} is a pointer; "
                            "did you mean to use '->'?",
                            TypeDescription(lhs_type)),
              location);
      return std::make_unique<DILErrorNode>(bad_type);
    }
  }

  // Check if LHS is a record type, i.e. class/struct or union.
  if (!IsRecordType(lhs_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(
                "member reference base type {0} is not a structure or union",
                TypeDescription(lhs_type)),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  lldb::ValueObjectSP lhs_valobj_sp;
  const DILAstNode* ast_node = lhs.get();
  if (ast_node->what_am_i() == DILNodeKind::kIdentifierNode) {
    const IdentifierNode* id_node = static_cast<const IdentifierNode*>(ast_node);
    auto identifier = static_cast<const IdentifierInfo&>(id_node->info());
    lhs_valobj_sp = identifier.value();
  }
  auto [member, idx] = GetMemberInfo(lhs_valobj_sp, lhs_type, member_id,
                                     UseSynthetic());

  if (!member) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("no member named '{0}' in {1}", member_id,
                          TypeDescription(GetUnqualifiedType(lhs_type))),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  uint32_t bitfield_size =
      member.is_bitfield ? member.bitfield_size_in_bits : 0;
  uint64_t byte_size = 0;
  if (auto temp =
      member.type.GetByteSize(m_ctx_scope.get()))
    byte_size = temp.value();
  if (bitfield_size > byte_size * CHAR_BIT) {
    // If the declared bitfield size is exceeding the type size, shrink
    // the bitfield size to the size of the type in bits.
    bitfield_size = byte_size * CHAR_BIT;
  }

  std::string tmp_name= "";
  if (member.name)
    tmp_name = member.name.value();
  ConstString field_name(tmp_name.c_str());
  return std::make_unique<MemberOfNode>(location, member.type, std::move(lhs),
                                        member.is_bitfield, bitfield_size,
                                        std::move(idx), is_arrow,
                                        member.is_synthetic, field_name);
}

void DILParser::Expect(clang::tok::TokenKind kind) {
  if (m_token.isNot(kind)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected {0}, got: {1}", TokenKindsJoin(kind),
                          TokenDescription(m_token)),
            m_token.getLocation());
  }
}

template <typename... Ts>
void DILParser::ExpectOneOf(clang::tok::TokenKind k, Ts... ks) {
  static_assert((std::is_same_v<Ts, clang::tok::TokenKind> && ...),
                "ExpectOneOf can be only called with values of type "
                "clang::tok::TokenKind");

  if (!m_token.isOneOf(k, ks...)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected any of ({0}), got: {1}",
                          TokenKindsJoin(k, ks...), TokenDescription(m_token)),
            m_token.getLocation());
  }
}

ExprResult DILParser::BuildIncrementDecrement(UnaryOpKind kind, ExprResult rhs,
                                              clang::SourceLocation location) {
  assert((kind == UnaryOpKind::PreInc || kind == UnaryOpKind::PreDec ||
          kind == UnaryOpKind::PostInc || kind == UnaryOpKind::PostDec) &&
         "illegal unary op kind, expected inc/dec");

  CompilerType bad_type;
  auto rhs_type = rhs->result_type_deref();

  // In C++ the requirement here is that the expression is "assignable". However
  // in the debugger context side-effects are not allowed and the only case
  // where increment/decrement are permitted is when modifying the "context
  // variable".
  // Technically, `++(++$var)` could be allowed too, since both increments
  // modify the context variable. However, MSVC debugger doesn't allow it, so we
  // don't implement it too.
  if (rhs->is_rvalue()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("expression is not assignable"), location);
    return std::make_unique<DILErrorNode>(bad_type);
  }
  if (!rhs->is_context_var() && !AllowSideEffects()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("side effects are not supported in this context: "
                          "trying to modify data at the target process"),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }
  llvm::StringRef op_name =
      (kind == UnaryOpKind::PreInc || kind == UnaryOpKind::PostInc)
          ? "increment"
          : "decrement";
  if (IsEnum(rhs_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot {0} expression of enum type '{1}'", op_name,
                          rhs_type.GetTypeName()),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }
  if (!IsScalar(rhs_type) && !IsPointerType(rhs_type)) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot {0} value of type '{1}'", op_name,
                          rhs_type.GetTypeName()),
            location);
    return std::make_unique<DILErrorNode>(bad_type);
  }

  return std::make_unique<UnaryOpNode>(location, rhs->result_type(), kind,
                                       std::move(rhs));
}

CompilerType DILParser::PrepareBinaryAddition(ExprResult& lhs, ExprResult& rhs,
                                              clang::SourceLocation location,
                                              bool is_comp_assign) {
  // Operation '+' works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  {integer,unscoped_enum} <-> pointer
  //  pointer <-> {integer,unscoped_enum}

  CompilerType bad_type;
  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (IsScalar(result_type)) {
    return result_type;
  }

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  // Check for pointer arithmetic operation.
  CompilerType ptr_type, integer_type;

  if (IsPointerType(lhs_type)) {
    ptr_type = lhs_type;
    integer_type = rhs_type;
  } else if (IsPointerType(rhs_type)) {
    integer_type = lhs_type;
    ptr_type = rhs_type;
  }

  if (!ptr_type || !IsInteger(integer_type)) {
    return bad_type;
  }

  if (IsPointerToVoid(ptr_type)) {
    BailOut(ErrorCode::kInvalidOperandType, "arithmetic on a pointer to void",
            location);
    return bad_type;
  }

  return ptr_type;
}

CompilerType DILParser::PrepareBinarySubtraction(ExprResult& lhs,
                                                 ExprResult& rhs,
                                                 clang::SourceLocation location,
                                                 bool is_comp_assign) {
  // Operation '-' works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  pointer <-> {integer,unscoped_enum}
  //  pointer <-> pointer (if pointee types are compatible)

  CompilerType bad_type;
  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (IsScalar(result_type)) {
    return result_type;
  }

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  if (IsPointerType(lhs_type) && IsInteger(rhs_type)) {
    if (IsPointerToVoid(lhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType, "arithmetic on a pointer to void",
              location);
      return bad_type;
    }

    return lhs_type;
  }

  if (IsPointerType(lhs_type) && IsPointerType(rhs_type)) {
    if (IsPointerToVoid(lhs_type) && IsPointerToVoid(rhs_type)) {
      BailOut(ErrorCode::kInvalidOperandType, "arithmetic on pointers to void",
              location);
      return bad_type;
    }

    // Compare canonical unqualified pointer types.
    CompilerType lhs_unqualified_type =
        lhs_type.GetCanonicalType().GetFullyUnqualifiedType();
    CompilerType rhs_unqualified_type =
        rhs_type.GetCanonicalType().GetFullyUnqualifiedType();
    bool comparable =
        CompareTypes(lhs_unqualified_type, rhs_unqualified_type);

    if (!comparable) {
      BailOut(
          ErrorCode::kInvalidOperandType,
          llvm::formatv("{0} and {1} are not pointers to compatible types",
                        TypeDescription(lhs_type), TypeDescription(rhs_type)),
          location);
      return bad_type;
    }

    // Pointer difference is ptrdiff_t.
    return GetBasicType(m_ctx_scope, GetPtrDiffType(m_ctx_scope));
  }

  // Invalid operands.
  return bad_type;
}

CompilerType DILParser::PrepareBinaryMulDiv(ExprResult& lhs, ExprResult& rhs,
                                            bool is_comp_assign) {
  // Operations {'*', '/'} work for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  // TODO(werat): Check for arithmetic zero division.
  if (IsScalar(result_type)) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;
}

CompilerType DILParser::PrepareBinaryRemainder(ExprResult& lhs, ExprResult& rhs,
                                               bool is_comp_assign) {
  // Operation '%' works for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  // TODO(werat): Check for arithmetic zero division.
  if (IsInteger(result_type)) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;;
}

CompilerType DILParser::PrepareBinaryBitwise(ExprResult& lhs, ExprResult& rhs,
                                             bool is_comp_assign) {
  // Operations {'&', '|', '^'} work for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}
  //
  // Note that {'<<', '>>'} are handled in a separate method.

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (IsInteger(result_type)) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;;
}

CompilerType DILParser::PrepareBinaryShift(ExprResult& lhs, ExprResult& rhs,
                                           bool is_comp_assign) {
  // Operations {'<<', '>>'} work for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}
  CompilerType bad_type;

  if (!is_comp_assign) {
    lhs = UsualUnaryConversions(m_ctx_scope, std::move(lhs));
  }
  rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  if (!IsInteger(lhs_type) || !IsInteger(rhs_type)) {
    return bad_type;
  }

  // The type of the result is that of the promoted left operand.
  return DoIntegralPromotion(m_ctx_scope, lhs_type);
}

CompilerType DILParser::PrepareBinaryComparison(BinaryOpKind kind,
                                                ExprResult& lhs,
                                                ExprResult& rhs,
                                                clang::SourceLocation location)
{
  // Comparison works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  scoped_enum <-> scoped_enum (if the same type)
  //  pointer <-> pointer (if pointee types are compatible)
  //  pointer <-> {integer,unscoped_enum,nullptr_t}
  //  {integer,unscoped_enum,nullptr_t} <-> pointer
  //  nullptr_t <-> {nullptr_t,integer} (if integer is literal zero)
  //  {nullptr_t,integer} <-> nullptr_t (if integer is literal zero)

  // If the operands has arithmetic or enumeration type (scoped or unscoped),
  // usual arithmetic conversions are performed on both operands following the
  // rules for arithmetic operators.
  CompilerType bad_type;
  auto _ = UsualArithmeticConversions(m_ctx_scope, lhs, rhs);

  // Apply smart-pointer-to-pointer conversions.
  if (IsSmartPtrType(lhs->result_type_deref())) {
    lhs = InsertSmartPtrToPointerConversion(std::move(lhs));
  }
  if (IsSmartPtrType(rhs->result_type_deref())) {
    rhs = InsertSmartPtrToPointerConversion(std::move(rhs));
  }

  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  // The result of the comparison is always bool.
  auto boolean_ty = GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);

  if (IsScalarOrUnscopedEnum(lhs_type) &&
      IsScalarOrUnscopedEnum(rhs_type)) {
    return boolean_ty;
  }

  // Scoped enums can be compared only to the instances of the same type.
  if (IsScopedEnum(lhs_type) || IsScopedEnum(rhs_type)) {
    if (CompareTypes(lhs_type, rhs_type)) {
      return boolean_ty;
    }
    // Invalid operands.
    return bad_type;;
  }

  bool is_ordered = (kind == BinaryOpKind::LT || kind == BinaryOpKind::LE ||
                     kind == BinaryOpKind::GT || kind == BinaryOpKind::GE);

  // Check if the value can be compared to a pointer. We allow all pointers,
  // integers, unscoped enumerations and a nullptr literal if it's an
  // equality/inequality comparison. For "pointer <-> integer" C++ allows only
  // equality/inequality comparison against literal zero and nullptr. However in
  // the debugger context it's often useful to compare a pointer with an integer
  // representing an address. That said, this also allows comparing nullptr and
  // any integer, not just literal zero, e.g. "nullptr == 1 -> false". C++
  // doesn't allow it, but we implement this for convenience.
  auto comparable_to_pointer = [&](CompilerType t) {
    return IsPointerType(t) || IsInteger(t) || IsUnscopedEnum(t) ||
           (!is_ordered && IsNullPtrType(t));
  };

  if ((IsPointerType(lhs_type) && comparable_to_pointer(rhs_type)) ||
      (comparable_to_pointer(lhs_type) && IsPointerType(rhs_type))) {
    // If both are pointers, check if they have comparable types. Comparing
    // pointers to void is always allowed.
    if ((IsPointerType(lhs_type) && !IsPointerToVoid(lhs_type)) &&
        (IsPointerType(rhs_type) && !IsPointerToVoid(rhs_type))) {
      // Compare canonical unqualified pointer types.
      CompilerType lhs_unqualified_type =
          lhs_type.GetCanonicalType().GetFullyUnqualifiedType();
      CompilerType rhs_unqualified_type =
          rhs_type.GetCanonicalType().GetFullyUnqualifiedType();
      bool comparable =
          CompareTypes(lhs_unqualified_type, rhs_unqualified_type);

      if (!comparable) {
        BailOut(
            ErrorCode::kInvalidOperandType,
            llvm::formatv("comparison of distinct pointer types ({0} and {1})",
                          TypeDescription(lhs_type), TypeDescription(rhs_type)),
            location);
        return bad_type;;
      }
    }

    return boolean_ty;
  }

  bool lhs_nullptr_or_zero =
      IsNullPtrType(lhs_type) || lhs->is_literal_zero();
  bool rhs_nullptr_or_zero =
      IsNullPtrType(rhs_type) || rhs->is_literal_zero();

  if (!is_ordered && ((IsNullPtrType(lhs_type) && rhs_nullptr_or_zero) ||
                      (lhs_nullptr_or_zero && IsNullPtrType(rhs_type)))) {
    return boolean_ty;
  }

  // Invalid operands.
  return bad_type;
}

CompilerType DILParser::PrepareBinaryLogical(const ExprResult& lhs,
                                             const ExprResult& rhs) {
  CompilerType bad_type;
  auto lhs_type = lhs->result_type_deref();
  auto rhs_type = rhs->result_type_deref();

  if (!IsContextuallyConvertibleToBool(lhs_type)) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, TypeDescription(lhs_type)),
        lhs->location());
    return bad_type;
  }

  if (!IsContextuallyConvertibleToBool(rhs_type)) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, TypeDescription(rhs_type)),
        rhs->location());
    return bad_type;
  }

  // The result of the logical operator is always bool.
  return GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);
}

CompilerType DILParser::PrepareCompositeAssignment(
    CompilerType comp_assign_type,
    const ExprResult& lhs,
    clang::SourceLocation location) {
  // In C++ the requirement here is that the expression is "assignable".
  // However in the debugger context side-effects are not allowed and the only
  // case where composite assignments are permitted is when modifying the
  // "context variable".
  // Technically, `($var += 1) += 1` could be allowed too, since both
  // operations modify the context variable. However, MSVC debugger doesn't
  // allow it, so we don't implement it too.
  CompilerType bad_type;
  if (lhs->is_rvalue()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("expression is not assignable"), location);
    return bad_type;
  }
  if (!lhs->is_context_var() && !AllowSideEffects()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("side effects are not supported in this context: "
                          "trying to modify data at the target process"),
            location);
    return bad_type;
  }

  // Check if we can assign the result of the binary operation back to LHS.
  auto lhs_type = lhs->result_type_deref();

  if (CompareTypes(comp_assign_type, lhs_type) ||
      ImplicitConversionIsAllowed(comp_assign_type, lhs_type)) {
    return lhs_type;
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("no known conversion from {0} to {1}",
                        TypeDescription(comp_assign_type),
                        TypeDescription(lhs_type)),
          location);
  return bad_type;
}

void DILParser::BailOut(ErrorCode code, const std::string& error,
                        clang::SourceLocation loc) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }

  m_error.SetError ((uint32_t) code, lldb::eErrorTypeGeneric);
  //  m_error.SetErrorString(FormatDiagnostics(m_smff, error, loc));
  m_error.SetErrorString(FormatDiagnostics(m_sm->GetSourceManager(), error,
                                           loc));
  m_token.setKind(clang::tok::eof);
}

void DILParser::ConsumeToken() {
  if (m_token.is(clang::tok::eof)) {
    // Don't do anything if we're already at eof. This can happen if an error
    // occurred during parsing and we're trying to bail out.
    return;
  }
  m_pp->Lex(m_token);
}

std::string DILParser::TokenDescription(const clang::Token& token) {
  const auto& spelling = m_pp->getSpelling(token);
  const auto* kind_name = token.getName();
  return llvm::formatv("<'{0}' ({1})>", spelling, kind_name);
}

std::string DILParser::FormatDiagnostics(
    clang::SourceManager& sm,
    const std::string& message,
    clang::SourceLocation loc) {
  // Get the source buffer and the location of the current token.
  llvm::StringRef text = sm.getBufferData(sm.getFileID(loc));
  size_t loc_offset = sm.getCharacterData(loc) - text.data();

  // Look for the start of the line.
  size_t line_start = text.rfind('\n', loc_offset);
  line_start = line_start == llvm::StringRef::npos ? 0 : line_start + 1;

  // Look for the end of the line.
  size_t line_end = text.find('\n', loc_offset);
  line_end = line_end == llvm::StringRef::npos ? text.size() : line_end;

  // Get a view of the current line in the source code and the position of the
  // diagnostics pointer.
  llvm::StringRef line = text.slice(line_start, line_end);
  int32_t arrow = sm.getPresumedColumnNumber(loc);

  // Calculate the padding in case we point outside of the expression (this can
  // happen if the parser expected something, but got EOF).˚
  size_t expr_rpad = std::max(0, arrow - static_cast<int32_t>(line.size()));
  size_t arrow_rpad = std::max(0, static_cast<int32_t>(line.size()) - arrow);

  return llvm::formatv("{0}: {1}\n{2}\n{3}", loc.printToString(sm),
                       message, llvm::fmt_pad(line, 0, expr_rpad),
                       llvm::fmt_pad("^", arrow - 1, arrow_rpad));
}

bool DILParser::ImplicitConversionIsAllowed(CompilerType src, CompilerType dst,
                                            bool is_src_literal_zero) {
  if (IsInteger(dst) || IsFloat(dst)) {
    // Arithmetic types and enumerations can be implicitly converted to integers
    // and floating point types.
    if (IsScalarOrUnscopedEnum(src) || IsScopedEnum(src)) {
      return true;
    }
  }

  if (IsPointerType(dst)) {
    // Literal zero, `nullptr_t` and arrays can be implicitly converted to
    // pointers.
    if (is_src_literal_zero || IsNullPtrType(src)) {
      return true;
    }
    if (IsArrayType(src) &&
        CompareTypes(src.GetArrayElementType(nullptr), GetPointeeType(dst))) {
      return true;
    }
  }

  return false;
}

ExprResult DILParser::InsertImplicitConversion(ExprResult expr,
                                               CompilerType type) {
  auto expr_type = expr->result_type_deref();

  // If the expression already has the required type, nothing to do here.
  if (CompareTypes(expr_type, type)) {
    return expr;
  }

  // Check if the implicit conversion is possible and insert a cast.
  if (ImplicitConversionIsAllowed(expr_type, type, expr->is_literal_zero())) {
    if (type.GetCanonicalType().GetBasicTypeEnumeration() !=
        lldb::eBasicTypeInvalid) {
      return std::make_unique<CStyleCastNode>(
          expr->location(), type, std::move(expr), CStyleCastKind::kArithmetic);
    }

    if (IsPointerType(type)) {
      return std::make_unique<CStyleCastNode>(
          expr->location(), type, std::move(expr), CStyleCastKind::kPointer);
    }

    // TODO(werat): What about if the conversion is not `kArithmetic` or
    // `kPointer`?
    llvm_unreachable("invalid implicit cast kind");
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("no known conversion from {0} to {1}",
                        TypeDescription(expr_type), TypeDescription(type)),
          expr->location());
  CompilerType bad_type;
  return std::make_unique<DILErrorNode>(bad_type);
}


lldb::BasicType TypeDeclaration::GetBasicType() const {
  assert(m_is_builtin && "type declaration doesn't describe a builtin type");

  if (m_sign_specifier == SignSpecifier::kSigned &&
      m_type_specifier == TypeSpecifier::kChar) {
    // "signed char" isn't the same as "char".
    return lldb::eBasicTypeSignedChar;
  }

  if (m_sign_specifier == SignSpecifier::kUnsigned) {
    switch (m_type_specifier) {
        // clang-format off
      // "unsigned" is "unsigned int"
      case TypeSpecifier::kUnknown:  return lldb::eBasicTypeUnsignedInt;
      case TypeSpecifier::kChar:     return lldb::eBasicTypeUnsignedChar;
      case TypeSpecifier::kShort:    return lldb::eBasicTypeUnsignedShort;
      case TypeSpecifier::kInt:      return lldb::eBasicTypeUnsignedInt;
      case TypeSpecifier::kLong:     return lldb::eBasicTypeUnsignedLong;
      case TypeSpecifier::kLongLong: return lldb::eBasicTypeUnsignedLongLong;
      // clang-format on
      default:
        assert(false && "unknown unsigned basic type");
        return lldb::eBasicTypeInvalid;
    }
  }

  switch (m_type_specifier) {
      // clang-format off
    case TypeSpecifier::kUnknown:
      // "signed" is "signed int"
      assert(m_sign_specifier == SignSpecifier::kSigned &&
             "invalid basic type declaration");
      return lldb::eBasicTypeInt;
    case TypeSpecifier::kVoid:       return lldb::eBasicTypeVoid;
    case TypeSpecifier::kBool:       return lldb::eBasicTypeBool;
    case TypeSpecifier::kChar:       return lldb::eBasicTypeChar;
    case TypeSpecifier::kShort:      return lldb::eBasicTypeShort;
    case TypeSpecifier::kInt:        return lldb::eBasicTypeInt;
    case TypeSpecifier::kLong:       return lldb::eBasicTypeLong;
    case TypeSpecifier::kLongLong:   return lldb::eBasicTypeLongLong;
    case TypeSpecifier::kFloat:      return lldb::eBasicTypeFloat;
    case TypeSpecifier::kDouble:     return lldb::eBasicTypeDouble;
    case TypeSpecifier::kLongDouble: return lldb::eBasicTypeLongDouble;
    case TypeSpecifier::kWChar:      return lldb::eBasicTypeWChar;
    case TypeSpecifier::kChar16:     return lldb::eBasicTypeChar16;
    case TypeSpecifier::kChar32:     return lldb::eBasicTypeChar32;
      // clang-format on
  }

  return lldb::eBasicTypeInvalid;
}



}  // namespace lldb_private
