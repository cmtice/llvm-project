//===-- DILEval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/DILEval.h"

#include <memory>

#include "clang/Basic/TokenKinds.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Target/DILAst.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

namespace lldb_private {

template <typename T>
bool Compare(BinaryOpKind kind, const T& l, const T& r) {
  switch (kind) {
    case BinaryOpKind::EQ:
      return l == r;
    case BinaryOpKind::NE:
      return l != r;
    case BinaryOpKind::LT:
      return l < r;
    case BinaryOpKind::LE:
      return l <= r;
    case BinaryOpKind::GT:
      return l > r;
    case BinaryOpKind::GE:
      return l >= r;

    default:
      assert(false && "invalid ast: invalid comparison operation");
      return false;
  }
}

static lldb::ValueObjectSP CreateValueFromBytes (lldb::TargetSP target_sp,
                                                 const void* bytes,
                                                 CompilerType type) {
  ExecutionContext exe_ctx(
      ExecutionContextRef(ExecutionContext(target_sp.get(), false)));
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target_sp.get()))
    byte_size = temp.value();
  lldb::DataExtractorSP data_sp =
      std::make_shared<DataExtractor> (
          bytes, byte_size,
          target_sp->GetArchitecture().GetByteOrder(),
          static_cast<uint8_t>(target_sp->GetArchitecture().GetAddressByteSize()));
  lldb::ValueObjectSP value =
      ValueObject::CreateValueObjectFromData("$result", *data_sp, exe_ctx,
                                             type);
  return value;

}

static lldb::ValueObjectSP CreateValueFromBytes (lldb::TargetSP target,
                                                 const void* bytes,
                                                 lldb::BasicType type) {
  CompilerType target_type;
  if (target) {
    for (auto type_system_sp : target->GetScratchTypeSystems())
      if (auto compiler_type = type_system_sp->GetBasicTypeFromAST(type)) {
        target_type = compiler_type;
        break;
      }
  }
  return CreateValueFromBytes(target, bytes, target_type);
}

static lldb::ValueObjectSP CreateValueFromAPInt (lldb::TargetSP target,
                                                 const llvm::APInt &v,
                                                 CompilerType type) {
  return CreateValueFromBytes(target, v.getRawData(), type);
}

static lldb::ValueObjectSP CreateValueFromAPFloat (lldb::TargetSP target,
                                                   const llvm::APFloat& v,
                                                   CompilerType type) {
  return CreateValueFromAPInt(target, v.bitcastToAPInt(), type);
}

static lldb::ValueObjectSP CreateValueFromPointer (lldb::TargetSP target,
                                                uintptr_t addr,
                                                CompilerType type) {
  return CreateValueFromBytes(target, &addr, type);
}

static lldb::ValueObjectSP CreateValueFromBool (lldb::TargetSP target,
                                                bool value) {
  return CreateValueFromBytes(target, &value, lldb::eBasicTypeBool);
}


static lldb::ValueObjectSP CreateValueNullptr(lldb::TargetSP target,
                                              CompilerType type) {
  assert(IsNullPtrType(type) && "target type must be nullptr");
  uintptr_t zero = 0;
  return CreateValueFromBytes(target, &zero, type);
}

static bool IsBasicType(CompilerType type) {
  return type.GetCanonicalType().GetBasicTypeEnumeration()
      != lldb::eBasicTypeInvalid;
}

static bool IsContextuallyConvertibleToBool(CompilerType type) {
  return IsScalar(type) || IsUnscopedEnum(type) || IsPointerType(type) ||
      IsNullPtrType(type) || IsArrayType(type);
}

static llvm::APFloat CreateAPFloatFromAPSInt(const llvm::APSInt& value,
                                             lldb::BasicType basic_type) {
  switch (basic_type) {
    case lldb::eBasicTypeFloat:
      return llvm::APFloat(value.isSigned()
                               ? llvm::APIntOps::RoundSignedAPIntToFloat(value)
                               : llvm::APIntOps::RoundAPIntToFloat(value));
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble:
      return llvm::APFloat(value.isSigned()
                               ? llvm::APIntOps::RoundSignedAPIntToDouble(value)
                               : llvm::APIntOps::RoundAPIntToDouble(value));
    default:
      return llvm::APFloat(NAN);
  }
}

static llvm::APFloat CreateAPFloatFromAPFloat(llvm::APFloat value,
                                              lldb::BasicType basic_type) {
  switch (basic_type) {
    case lldb::eBasicTypeFloat: {
      bool loses_info;
      value.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
      return value;
    }
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble: {
      bool loses_info;
      value.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
      return value;
    }
    default:
      return llvm::APFloat(NAN);
  }
}

static uint64_t GetByteSize(CompilerType type, lldb::TargetSP target) {
  ExecutionContext exe_ctx(
      ExecutionContextRef(ExecutionContext(target.get(), false)));
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    byte_size = temp.value();

  return byte_size;
}

static llvm::APSInt GetInteger(lldb::ValueObjectSP value_sp)
{
  lldb::TargetSP target = value_sp->GetTargetSP();
  unsigned bit_width =
      static_cast<unsigned>(
          GetByteSize(value_sp->GetCompilerType(), target) * CHAR_BIT);
  lldb::ValueObjectSP value(DILGetSPWithLock(value_sp));
  bool success = true;
  uint64_t fail_value = 0;
  uint64_t ret_val = value->GetValueAsUnsigned(fail_value, &success);
  uint64_t new_value = fail_value;
  if (success)
    new_value = ret_val;
  bool is_signed = IsSigned(value->GetCompilerType());

  return llvm::APSInt(llvm::APInt(bit_width, new_value, is_signed), !is_signed);
}


static llvm::APFloat GetFloat(lldb::ValueObjectSP value) {
  lldb::BasicType basic_type =
      value->GetCompilerType().GetCanonicalType().GetBasicTypeEnumeration();
  lldb::DataExtractorSP data_sp(new DataExtractor());
  Status error;

  switch (basic_type) {
    case lldb::eBasicTypeFloat: {
      float v = 0;
      value->GetData(*data_sp, error);
      assert (error.Success() && "Unable to read float data from value");

      lldb::offset_t offset = 0;
      uint32_t old_offset = offset;
      void *ok = nullptr;
      ok = data_sp->GetU8(&offset, (void *) &v, sizeof(float));
      assert(offset != old_offset && ok != nullptr && "unable to read data");

      return llvm::APFloat(v);
    }
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble: {
      double v = 0;
      value->GetData(*data_sp, error);
      assert (error.Success() && "Unable to read long double data from value");

      lldb::offset_t offset = 0;
      uint32_t old_offset = offset;
      void *ok = nullptr;
      ok = data_sp->GetU8(&offset, (void *) &v, sizeof(double));
      assert(offset != old_offset && ok != nullptr && "unable to read data");

      return llvm::APFloat(v);
    }
    default:
      return llvm::APFloat(NAN);
  }
}

static uint64_t GetUInt64(lldb::ValueObjectSP value_sp) {
  // GetValueAsUnsigned performs overflow according to the underlying type. For
  // example, if the underlying type is `int32_t` and the value is `-1`,
  // GetValueAsUnsigned will return 4294967295.
  lldb::ValueObjectSP value(DILGetSPWithLock(value_sp));
  return IsSigned(value->GetCompilerType())
      ? value->GetValueAsSigned(0)
      : value->GetValueAsUnsigned(0);
}

static bool GetBool(lldb::ValueObjectSP value) {
  CompilerType val_type = value->GetCompilerType();
  if (IsInteger(val_type) || IsUnscopedEnum(val_type) ||
      IsPointerType(val_type)) {
    return GetInteger(value).getBoolValue();
  }
  if (IsFloat(val_type)) {
    return GetFloat(value).isNonZero();
  }
  if (IsArrayType(val_type)) {
    lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
    lldb::ValueObjectSP new_val =
        ValueObject::CreateValueObjectFromAddress(
            value_sp->GetName().GetStringRef(),
            value_sp->GetAddressOf(),
            value_sp->GetExecutionContextRef(),
            val_type);
    return GetUInt64(new_val) != 0;
  }
  return false;
}

static lldb::ValueObjectSP Clone(lldb::ValueObjectSP val) {
  lldb::DataExtractorSP data_sp(new DataExtractor());
  Status error;
  val->GetData(*data_sp, error);
  if (error.Success()) {
    Status ignore;
    void *ok = nullptr;
    auto raw_data = std::make_unique<uint8_t[]>(data_sp->GetByteSize());
    lldb::offset_t offset = 0;
    uint32_t old_offset = offset;
    size_t size = data_sp->GetByteSize();
    ok = data_sp->GetU8(&offset, raw_data.get(), size);
    if ((offset == old_offset) || (ok == nullptr)) {
      ignore.SetErrorString("Clone: unable to read data");
      return lldb::ValueObjectSP();
    }
    return CreateValueFromBytes(val->GetTargetSP(), raw_data.get(),
                                val->GetCompilerType());
  } else {
    return lldb::ValueObjectSP();
  }
}

static void Update(lldb::ValueObjectSP val, const llvm::APInt& v) {
  assert(v.getBitWidth() == GetByteSize(val->GetCompilerType(),
                                        val->GetTargetSP()) * CHAR_BIT &&
         "illegal argument: new value should be of the same size");

  lldb::DataExtractorSP data_sp;
  Status error;
  lldb::TargetSP target = val->GetTargetSP();
  data_sp->SetData(v.getRawData(),
                   GetByteSize(val->GetCompilerType(), target),
                   target->GetArchitecture().GetByteOrder());
  data_sp->SetAddressByteSize(
      static_cast<uint8_t>(target->GetArchitecture().GetAddressByteSize()));
  val->SetData(*data_sp, error);
}

static void Update(lldb::ValueObjectSP val, lldb::ValueObjectSP val2) {
  CompilerType val2_type = val2->GetCompilerType();
  assert((IsInteger(val2_type) || IsFloat(val2_type) ||
          IsPointerType(val2_type)) &&
         "illegal argument: new value should be of the same size");

  if (IsInteger(val2_type)) {
    Update(val, GetInteger(val2));
  } else if (IsFloat(val2_type)) {
    Update(val, GetFloat(val2).bitcastToAPInt());
  } else if (IsPointerType(val2_type)) {
    Update(val, llvm::APInt(64, GetUInt64(val2)));
  }
}


static lldb::ValueObjectSP EvaluateArithmeticOpInteger(lldb::TargetSP target,
                                                       BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs,
                                                       CompilerType rtype)
{
  assert(IsInteger(lhs->GetCompilerType()) &&
         IsInteger(rhs->GetCompilerType()) &&
         "invalid ast: both operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr ||
          CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must have the same type");

  auto wrap = [target, rtype](auto value) {
    return CreateValueFromAPInt(target, value, rtype);
  };

  auto l = GetInteger(lhs);
  auto r = GetInteger(rhs);

  switch (kind) {
    case BinaryOpKind::Add:
      return wrap(l + r);
    case BinaryOpKind::Sub:
      return wrap(l - r);
    case BinaryOpKind::Div:
      return wrap(l / r);
    case BinaryOpKind::Mul:
      return wrap(l * r);
    case BinaryOpKind::Rem:
      return wrap(l % r);
    case BinaryOpKind::And:
      return wrap(l & r);
    case BinaryOpKind::Or:
      return wrap(l | r);
    case BinaryOpKind::Xor:
      return wrap(l ^ r);
    case BinaryOpKind::Shl:
      return wrap(l.shl(r));
    case BinaryOpKind::Shr:
      // Apply arithmetic shift on signed values and logical shift operation
      // on unsigned values.
      return wrap(l.isSigned() ? l.ashr(r) : l.lshr(r));

    default:
      assert(false && "invalid ast: invalid arithmetic operation");
      return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOpFloat(lldb::TargetSP target,
                                                     BinaryOpKind kind,
                                                     lldb::ValueObjectSP lhs,
                                                     lldb::ValueObjectSP rhs,
                                                     CompilerType rtype) {
  assert((IsFloat(lhs->GetCompilerType()) &&
          CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must be floats and have the same type");

  auto wrap = [target, rtype](auto value) {
    return CreateValueFromAPFloat(target, value, rtype);
  };

  auto l = GetFloat(lhs);
  auto r = GetFloat(rhs);

  switch (kind) {
    case BinaryOpKind::Add:
      return wrap(l + r);
    case BinaryOpKind::Sub:
      return wrap(l - r);
    case BinaryOpKind::Div:
      return wrap(l / r);
    case BinaryOpKind::Mul:
      return wrap(l * r);

    default:
      assert(false && "invalid ast: invalid arithmetic operation");
      return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOp(lldb::TargetSP target,
                                                BinaryOpKind kind,
                                                lldb::ValueObjectSP lhs,
                                                lldb::ValueObjectSP rhs,
                                                CompilerType rtype) {
  assert((IsInteger(rtype) || IsFloat(rtype)) &&
         "invalid ast: result type must either integer or floating point");

  // Evaluate arithmetic operation for two integral values.
  if (IsInteger(rtype)) {
    return EvaluateArithmeticOpInteger(target, kind, lhs, rhs, rtype);
  }

  // Evaluate arithmetic operation for two floating point values.
  if (IsFloat(rtype)) {
    return EvaluateArithmeticOpFloat(target, kind, lhs, rhs, rtype);
  }

  return lldb::ValueObjectSP();
}

static bool IsInvalidDivisionByMinusOne(lldb::ValueObjectSP lhs_sp,
                                        lldb::ValueObjectSP rhs_sp)
{
  assert(IsInteger(lhs_sp->GetCompilerType()) &&
         IsInteger(rhs_sp->GetCompilerType()) && "operands should be integers");

  lldb::ValueObjectSP rhs(DILGetSPWithLock(rhs_sp));
  lldb::ValueObjectSP lhs(DILGetSPWithLock(lhs_sp));
  // The result type should be signed integer.
  auto basic_type =
      rhs->GetCompilerType().GetCanonicalType().GetBasicTypeEnumeration();
  if (basic_type != lldb::eBasicTypeInt && basic_type != lldb::eBasicTypeLong &&
      basic_type != lldb::eBasicTypeLongLong) {
    return false;
  }

  // The RHS should be equal to -1.
  if (rhs->GetValueAsSigned(0) != -1) {
    return false;
  }

  // The LHS should be equal to the minimum value the result type can hold.
  auto bit_size = GetByteSize(rhs->GetCompilerType(), rhs->GetTargetSP()) *
                  CHAR_BIT;
  return lhs->GetValueAsSigned(0) + (1LLU << (bit_size - 1)) == 0;
}

static lldb::ValueObjectSP EvaluateMemberOf(lldb::ValueObjectSP value,
                                            const std::vector<uint32_t>& path,
                                            bool use_synthetic) {
  // The given `value` can be a pointer, but GetChildAtIndex works for pointers
  // too, so we don't need to dereference it explicitely. This also avoid having
  // an "ephemeral" parent lldb::ValueObjectSP, representing the dereferenced
  // value.
  lldb::ValueObjectSP member_val_sp = value;
  // Objects from the standard library (e.g. containers, smart pointers) have
  // synthetic children (e.g. stored values for containers, wrapped object for
  // smart pointers), but the indexes in `member_index()` array refer to the
  // actual type members.
  // bool use_synthetic = false;
  lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues;
  lldb::ValueObjectSP member_val(DILGetSPWithLock(member_val_sp, use_dynamic,
                                                  use_synthetic));
  for (uint32_t idx : path) {
    // Force static value, otherwise we can end up with the "real" type.
    member_val = member_val->GetChildAtIndex(idx, /*can_create*/ true);
  }
  assert(member_val && "invalid ast: invalid member access");

  // If value is a reference, derefernce it to get to the underlying type. All
  // operations on a reference should be actually operations on the referent.
  Status error;
  if (member_val->GetCompilerType().IsReferenceType()) {
    member_val = member_val->Dereference(error);
    assert(member_val && error.Success() && "unable to dereference member val");
  }

  return member_val;
}

static lldb::addr_t GetLoadAddress(lldb::ValueObjectSP inner_value_sp) {
  lldb::addr_t addr_value = LLDB_INVALID_ADDRESS;
  lldb::TargetSP target_sp(inner_value_sp->GetTargetSP());
  lldb::ValueObjectSP inner_value(DILGetSPWithLock(inner_value_sp));
  if (target_sp) {
    const bool scalar_is_load_address = true;
    AddressType addr_type;
    addr_value = inner_value->GetAddressOf(scalar_is_load_address, &addr_type);
    if (addr_type == eAddressTypeFile) {
      lldb::ModuleSP module_sp(inner_value->GetModule());
      if (!module_sp)
        addr_value = LLDB_INVALID_ADDRESS;
      else {
        Address tmp_addr;
        module_sp->ResolveFileAddress(addr_value, tmp_addr);
        addr_value = tmp_addr.GetLoadAddress(target_sp.get());
      }
    } else if (addr_type == eAddressTypeHost ||
               addr_type == eAddressTypeHost)
      addr_value = LLDB_INVALID_ADDRESS;
  }
  return addr_value;
}

static lldb::ValueObjectSP CastDerivedToBaseType(
    lldb::TargetSP target,
    lldb::ValueObjectSP value,
    CompilerType type,
    const std::vector<uint32_t>& idx)
{
  assert((IsPointerType(type) || type.IsReferenceType()) &&
         "invalid ast: target type should be a pointer or a reference");
  assert(!idx.empty() && "invalid ast: children sequence should be non-empty");

  // The `value` can be a pointer, but GetChildAtIndex works for pointers too.
  //bool use_synthetic_value = false;
  bool prefer_synthetic_value = false;
  lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues;
  lldb::ValueObjectSP inner_value(DILGetSPWithLock(value, use_dynamic,
                                                   prefer_synthetic_value));
  for (const uint32_t i : idx) {
    // Force static value, otherwise we can end up with the "real" type.
    inner_value = value->GetChildAtIndex(i, /*can_create_synthetic*/ false);
  }

  // At this point type of `inner_value` should be the dereferenced target type.
  CompilerType inner_value_type = inner_value->GetCompilerType();
  if (IsPointerType(type)) {
    assert(CompareTypes(inner_value_type, type.GetPointeeType()) &&
           "casted value doesn't match the desired type");

    uintptr_t addr = GetLoadAddress(inner_value);
    return CreateValueFromPointer(target, addr, type);
  }

  // At this point the target type should be a reference.
  assert(CompareTypes(inner_value_type, type.GetNonReferenceType()) &&
         "casted value doesn't match the desired type");

  lldb::ValueObjectSP inner_value_sp(DILGetSPWithLock(inner_value));
  return lldb::ValueObjectSP(inner_value_sp->Cast(type.GetNonReferenceType()));
}

static lldb::ValueObjectSP CastBaseToDerivedType(lldb::TargetSP target,
                                                 lldb::ValueObjectSP value,
                                                 CompilerType type,
                                                 uint64_t offset)
{
  assert((IsPointerType(type) || type.IsReferenceType()) &&
         "invalid ast: target type should be a pointer or a reference");

  auto pointer_type = IsPointerType(type)
                          ? type
                          : type.GetNonReferenceType().GetPointerType();

  uintptr_t addr = IsPointerType(type) ? GetUInt64(value)
                                         : GetLoadAddress(value);

  value = CreateValueFromPointer(target, addr - offset, pointer_type);

  if (IsPointerType(type)) {
    return value;
  }

  // At this point the target type is a reference. Since `value` is a pointer,
  // it has to be dereferenced.
  Status error;
  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  return value_sp->Dereference(error);
}

static std::string FormatDiagnostics(clang::SourceManager& sm,
                                     const std::string& message,
                                     clang::SourceLocation loc,
                                     ErrorCode code)
{
  const char *ecode_names[7] = {
    "kOK", "kInvalidExpressionSyntax", "kInvalidNumericLiteral",
    "kInvalidOperandType", "kUndeclaredIdentifier", "kNotImplemented",
    "kUnknown"};

  // Translate ErrorCode
  llvm::StringRef error_code = ecode_names[(int)code];

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
  // happen if the parser expected something, but got EOF).
  size_t expr_rpad = std::max(0, arrow - static_cast<int32_t>(line.size()));
  size_t arrow_rpad = std::max(0, static_cast<int32_t>(line.size()) - arrow);

  return llvm::formatv("{0}: {1}:{2}\n{3}\n{4}", loc.printToString(sm),
                       error_code, message,
                       llvm::fmt_pad(line, 0, expr_rpad),
                       llvm::fmt_pad("^", arrow - 1, arrow_rpad));
}

void SetUbStatus(Status& error, ErrorCode code) {
  llvm::StringRef err_str;
  switch ((int) code) {
    case (int) ErrorCode::kUBDivisionByZero:
      err_str ="Error: Division by zero detected.";
      break;
    case (int) ErrorCode::kUBDivisionByMinusOne:
      // If "a / b" isn't representable in its result type, then results of
      // "a / b" and "a % b" are undefined behaviour. This happens when "a"
      // is equal to the minimum value of the result type and "b" is equal
      // to -1.
      err_str ="Error: Invalid division by negative one  detected.";
      break;
    case (int) ErrorCode::kUBInvalidCast:
      err_str ="Error: Invalid type cast detected.";
      break;
    case (int) ErrorCode::kUBInvalidShift:
      err_str ="Error: Invalid shift detected.";
      break;
    case (int) ErrorCode::kUBNullPtrArithmetic:
      err_str ="Error: Attempt to perform arithmetic with null ptr  detected.";
      break;
    case (int) ErrorCode::kUBInvalidPtrDiff:
      err_str ="Error: Attempt to perform invalid ptr arithmetic detected.";
      break;
    default:
      err_str ="Error: Unknown undefined behavior error.";
      break;
  }
  error.SetError((lldb::ValueType)code, lldb::ErrorType::eErrorTypeGeneric);
  error.SetErrorString(err_str);
}


static lldb::ValueObjectSP CastScalarToBasicType(lldb::TargetSP target,
                                                 lldb::ValueObjectSP value,
                                                 CompilerType type,
                                                 Status& error)
{
  assert(IsScalar(type) && "target type must be an scalar");
  assert(IsScalar(value->GetCompilerType()) && "argument must be a scalar");

  if (IsBool(type)) {
    if (IsInteger(value->GetCompilerType())) {
      return CreateValueFromBool(target, GetUInt64(value) != 0);
    }
    if (IsFloat(value->GetCompilerType())) {
      return CreateValueFromBool(target, !GetFloat(value).isZero());
    }
  }
  if (IsInteger(type)) {
    if (IsInteger(value->GetCompilerType())) {
      llvm::APSInt ext =
          GetInteger(value).extOrTrunc(GetByteSize(type, target) * CHAR_BIT);
      return CreateValueFromAPInt(target, ext, type);
    }
    if (IsFloat(value->GetCompilerType())) {
      llvm::APSInt integer(GetByteSize(type, target) * CHAR_BIT, !IsSigned(type));
      bool is_exact;
      llvm::APFloatBase::opStatus status = GetFloat(value).convertToInteger(
          integer, llvm::APFloat::rmTowardZero, &is_exact);

      // Casting floating point values that are out of bounds of the target typ\
e
      // is undefined behaviour.
      if (status & llvm::APFloatBase::opInvalidOp) {
        SetUbStatus(error, ErrorCode::kUBInvalidCast);
      }

      return CreateValueFromAPInt(target, integer, type);
    }
  }
  if (IsFloat(type)) {
    if (IsInteger(value->GetCompilerType())) {
      llvm::APFloat f = CreateAPFloatFromAPSInt(
          GetInteger(value), type.GetCanonicalType().GetBasicTypeEnumeration());
      return CreateValueFromAPFloat(target, f, type);
    }
    if (IsFloat(value->GetCompilerType())) {
      llvm::APFloat f = CreateAPFloatFromAPFloat(
          GetFloat(value), type.GetCanonicalType().GetBasicTypeEnumeration());
      return CreateValueFromAPFloat(target, f, type);
    }
  }
  assert(false && "invalid target type: must be a scalar");
  return lldb::ValueObjectSP();
}

static lldb::ValueObjectSP CastEnumToBasicType(lldb::TargetSP target,
                                               lldb::ValueObjectSP val,
                                               CompilerType type)
{
  assert(IsScalar(type) && "target type must be a scalar");
  assert(IsEnum(val->GetCompilerType()) && "argument must be an enum");

  if (IsBool(type)) {
    return CreateValueFromBool(target, GetUInt64(val) != 0);
  }

  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      GetInteger(val).extOrTrunc(GetByteSize(type, target) * CHAR_BIT);

  if (IsInteger(type)) {
    return CreateValueFromAPInt(target, ext, type);
  }
  if (IsFloat(type)) {
    llvm::APFloat f =
        CreateAPFloatFromAPSInt(ext, type.GetCanonicalType().GetBasicTypeEnumeration());
    return CreateValueFromAPFloat(target, f, type);
  }
  assert(false && "invalid target type: must be a scalar");
  return lldb::ValueObjectSP();
}

static lldb::ValueObjectSP CastPointerToBasicType(lldb::TargetSP target,
                                                  lldb::ValueObjectSP val,
                                                  CompilerType type)
{
  assert(IsInteger(type) && "target type must be an integer");
  assert((IsBool(type) || (GetByteSize(type, target) >=
                           GetByteSize(val->GetCompilerType(), target)))
         && "target type cannot be smaller than the pointer type");

  if (IsBool(type)) {
    return CreateValueFromBool(target, GetUInt64(val) != 0);
  }

  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      GetInteger(val).extOrTrunc(GetByteSize(type, target) * CHAR_BIT);
  return CreateValueFromAPInt(target, ext, type);
}

static lldb::ValueObjectSP CastIntegerOrEnumToEnumType(lldb::TargetSP target,
                                                       lldb::ValueObjectSP val,
                                                       CompilerType type)
{
  assert(IsEnum(type) && "target type must be an enum");
  assert((IsInteger(val->GetCompilerType()) || IsEnum(val->GetCompilerType()))
         && "argument must be an integer or an enum");

  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      GetInteger(val).extOrTrunc(GetByteSize(type, target) * CHAR_BIT);
  return CreateValueFromAPInt(target, ext, type);
}

static lldb::ValueObjectSP CastFloatToEnumType(lldb::TargetSP target,
                                               lldb::ValueObjectSP val,
                                               CompilerType type,
                                               Status& error)
{
  assert(IsEnum(type) && "target type must be an enum");
  assert(IsFloat(val->GetCompilerType()) && "argument must be a float");

  llvm::APSInt integer(GetByteSize(type, target) * CHAR_BIT, !IsSigned(type));
  bool is_exact;

  llvm::APFloatBase::opStatus status = GetFloat(val).convertToInteger(
      integer, llvm::APFloat::rmTowardZero, &is_exact);

  // Casting floating point values that are out of bounds of the target type
  // is undefined behaviour.
  if (status & llvm::APFloatBase::opInvalidOp) {
    SetUbStatus(error, ErrorCode::kUBInvalidCast);
  }

  return CreateValueFromAPInt(target, integer, type);
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm)
    : m_target(std::move(target)), m_sm(std::move(sm))
{
  m_default_dynamic = lldb::eNoDynamicValues;
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::DynamicValueType use_dynamic)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_default_dynamic(use_dynamic) {}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::ValueObjectSP scope)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_scope(std::move(scope))
{
  m_default_dynamic = lldb::eNoDynamicValues;
  // If `m_scope` is a reference, dereference it. All operations on a reference
  // should be operations on the referent.
  if (m_scope->GetCompilerType().IsValid() &&
      m_scope->GetCompilerType().IsReferenceType()) {
    Status error;
    m_scope = m_scope->Dereference(error);
  }
}

void DILInterpreter::SetContextVars(
    std::unordered_map<std::string, lldb::ValueObjectSP> context_vars) {
  m_context_vars = std::move(context_vars);
}

lldb::ValueObjectSP DILInterpreter::DILEval(const DILAstNode* tree,
                                            lldb::TargetSP target_sp,
                                            Status& error)
{
  m_error.Clear();
  // Evaluate an AST.
  DILEvalNode(tree);
  // Set the error.
  error = m_error;
  // Return the computed result. If there was an error, it will be invalid.
  return m_result;
}

lldb::ValueObjectSP DILInterpreter::DILEvalNode(const DILAstNode* node,
                                             FlowAnalysis* flow) {
  // Set up the evaluation context for the current node.
  m_flow_analysis_chain.push_back(flow);
  // Traverse an AST pointed by the `node`.
  node->Accept(this);
  // Cleanup the context.
  m_flow_analysis_chain.pop_back();
  // Return the computed value for convenience. The caller is responsible for
  // checking if an error occured during the evaluation.
  return m_result;
}

void DILInterpreter::SetError(ErrorCode code, std::string error,
                              clang::SourceLocation loc) {
  assert(m_error.Success() && "interpreter can error only once");
  m_error.SetErrorString(
      FormatDiagnostics(m_sm->GetSourceManager(), error, loc, code));
}

void DILInterpreter::Visit(const DILErrorNode* node) {
  // The AST is not valid.
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const LiteralNode* node) {
  struct {
    lldb::ValueObjectSP operator()(llvm::APInt val) {
      return CreateValueFromAPInt(target, val, type);
    }
    lldb::ValueObjectSP operator()(llvm::APFloat val) {
      return CreateValueFromAPFloat(target, val, type);
    }
    lldb::ValueObjectSP operator()(bool val) {
      return CreateValueFromBool(target, val);
    }
    lldb::ValueObjectSP operator()(const std::vector<char>& val) {
      return CreateValueFromBytes(
          target, reinterpret_cast<const void*>(val.data()), type);
    }

    lldb::TargetSP target;
    CompilerType type;
  } visitor{m_target, node->result_type()};
  m_result = std::visit(visitor, node->value());
}

void DILInterpreter::Visit(const IdentifierNode* node) {
  auto identifier = static_cast<const IdentifierInfo&>(node->info());

  lldb::ValueObjectSP val;
  lldb::TargetSP target_sp;
  Status error;
  switch (identifier.kind()) {
    using Kind = IdentifierInfo::Kind;
    case Kind::kValue:
      val = identifier.value();
      target_sp = val->GetTargetSP();
      assert(target_sp && target_sp->IsValid() && "invalid ast: invalid identifier value");
      break;

    case Kind::kContextArg:
      assert(node->is_context_var() && "invalid ast: context var expected");
      val = ResolveContextVar(node->name());
      target_sp = val->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUndeclaredIdentifier,
            llvm::formatv("use of undeclared identifier '{0}'", node->name()),
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      if (!CompareTypes(node->result_type_deref(), val->GetCompilerType())) {
        SetError(ErrorCode::kInvalidOperandType,
                 llvm::formatv("unexpected type of context variable '{0}' "
                               "(expected {1}, got {2})",
                               node->name(),
                               TypeDescription(node->result_type_deref()),
                               TypeDescription(val->GetCompilerType())),
                 node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      break;

    case Kind::kMemberPath:
      target_sp = m_scope->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUnknown,
            llvm::formatv(
                "unable to resolve '{0}', evaluation requires a value context",
                node->name()),
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      val = EvaluateMemberOf(m_scope, identifier.path(), false);
      break;

    case Kind::kThisKeyword:
      target_sp = m_scope->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUnknown,
            "unable to resolve 'this', evaluation requires a value context",
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      val = m_scope->AddressOf(error);
      break;

    default:
      assert(false && "invalid ast: invalid identifier kind");
  }

  target_sp = val->GetTargetSP();
  assert(target_sp && target_sp->IsValid() &&
         "identifier doesn't resolve to a valid value");
  // TODO: Check that `val` type is matching the node's result type.

  // If value is a reference, dereference it to get to the underlying type. All
  // operations on a reference should be actually operations on the referent.
  if (val->GetCompilerType().IsReferenceType()) {
    // TODO(werat): LLDB canonizes the type upon a dereference. This looks like
    // a bug, but for now we need to mitigate it. Check if the resulting type is
    // incorrect and fix it up.
    // Not necessary if https://reviews.llvm.org/D103532 is available.
    auto deref_type = val->GetCompilerType().GetNonReferenceType();
    Status error;
    lldb::ValueObjectSP val_sp(DILGetSPWithLock(val));
    val = val_sp->Dereference(error);

    CompilerType val_type = val->GetCompilerType();
    if (val_type != deref_type) {
      val = lldb::ValueObjectSP(val->Cast(deref_type));
    }
  }

  m_result = val;
}

void DILInterpreter::Visit(const SizeOfNode* node) {
  auto operand = node->operand();

  // For reference type (int&) we need to look at the referenced type.
  size_t size = operand.IsReferenceType()
                ? GetByteSize(operand.GetNonReferenceType(), m_target)
                : GetByteSize(operand, m_target);
  CompilerType type = node->result_type_deref();
  m_result = CreateValueFromBytes(m_target, &size, type);
}

void DILInterpreter::Visit(const BuiltinFunctionCallNode* node) {
  if (node->name() == "__log2") {
    assert(node->arguments().size() == 1 &&
           "invalid ast: expected exactly one argument to `__log2`");
    // Get the first (and the only) argument and evaluate it.
    auto& arg = node->arguments()[0];
    lldb::ValueObjectSP val = DILEvalNode(arg.get());
    if (!val) {
      return;
    }
    assert(IsInteger(val->GetCompilerType()) &&
           "invalid ast: argument to __log2 must be an interger");

    // Use Log2_32 to match the behaviour of Visual Studio debugger.
    uint32_t ret = llvm::Log2_32(static_cast<uint32_t>(GetUInt64(val)));
    m_result = CreateValueFromBytes(m_target, &ret, lldb::eBasicTypeUnsignedInt);
    return;
  }

  if (node->name() == "__findnonnull") {
    assert(node->arguments().size() == 2 &&
           "invalid ast: expected exactly two arguments to `__findnonnull`");

    auto& arg1 = node->arguments()[0];
    lldb::ValueObjectSP val1_sp = DILEvalNode(arg1.get());
    if (!val1_sp) {
      return;
    }

    lldb::ValueObjectSP val1(DILGetSPWithLock(val1_sp));
    // Resolve data address for the first argument.
    uint64_t addr;

    if (IsPointerType(val1->GetCompilerType())) {
      addr = val1->GetValueAsUnsigned(0);
    } else if (val1->GetCompilerType().IsArrayType()) {
      addr = GetLoadAddress(val1);
    } else {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv("no known conversion from '{0}' to 'T*' for 1st "
                             "argument of __findnonnull()",
                             val1->GetCompilerType().GetTypeName()),
               arg1->location());
      return;
    }

    auto& arg2 = node->arguments()[1];
    lldb::ValueObjectSP val2 = DILEvalNode(arg2.get());
    if (!val2) {
      return;
    }
    lldb::ValueObjectSP val2_sp(DILGetSPWithLock(val2));
    int64_t size = val2_sp->GetValueAsSigned(0);

    if (size < 0 || size > 100000000) {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv(
                   "passing in a buffer size ('{0}') that is negative or in "
                   "excess of 100 million to __findnonnull() is not allowed.",
                   size),
               arg2->location());
      return;
    }

    lldb::ProcessSP process = m_target->GetProcessSP();
    size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize();

    uint64_t memory = 0;
    Status error;

    for (int i = 0; i < size; ++i) {
      size_t read =
          process->ReadMemory(addr + i * ptr_size, &memory, ptr_size, error);

      if (error.Fail() || read != ptr_size) {
        SetError(ErrorCode::kUnknown,
                 llvm::formatv("error calling __findnonnull(): {0}",
                               error.AsCString() ? error.AsCString()
                                                 : "cannot read memory"),
                 node->location());
        return;
      }

      if (memory != 0) {
        m_result = CreateValueFromBytes(m_target, &i, lldb::eBasicTypeInt);
        return;
      }
    }

    int ret = -1;
    m_result = CreateValueFromBytes(m_target, &ret, lldb::eBasicTypeInt);
    return;
  }

  assert(false && "invalid ast: unknown builtin function");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CStyleCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  switch (node->kind()) {
    case CStyleCastKind::kArithmetic: {
      assert((type.GetCanonicalType().GetBasicTypeEnumeration() !=
              lldb::eBasicTypeInvalid) &&
             "invalid ast: target type should be a basic type.");
      // Pick an appropriate cast.
      if (IsPointerType(rhs->GetCompilerType())
          || IsNullPtrType(rhs->GetCompilerType())) {
        m_result = CastPointerToBasicType(m_target, rhs, type);
      } else if (IsScalar(rhs->GetCompilerType())) {
        m_result = CastScalarToBasicType(m_target, rhs, type, m_error);
      } else if (IsEnum(rhs->GetCompilerType())) {
        m_result = CastEnumToBasicType(m_target, rhs, type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to arithmetic type");
      }
      return;
    }
    case CStyleCastKind::kEnumeration: {
      assert(IsEnum(type) &&
             "invalid ast: target type should be an enumeration.");

      if (IsFloat(rhs->GetCompilerType())) {
        m_result = CastFloatToEnumType(m_target, rhs, type, m_error);
      } else if (IsInteger(rhs->GetCompilerType()) ||
                 IsEnum(rhs->GetCompilerType())) {
        m_result = CastIntegerOrEnumToEnumType(m_target, rhs, type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }
    case CStyleCastKind::kPointer: {
      assert(IsPointerType(type) &&
             "invalid ast: target type should be a pointer.");
      uint64_t addr = rhs->GetCompilerType().IsArrayType()
                          ? GetLoadAddress(rhs)
                          : GetUInt64(rhs);
      m_result = CreateValueFromPointer(m_target, addr, type);
      return;
    }
    case CStyleCastKind::kNullptr: {
      assert((type.GetCanonicalType().GetBasicTypeEnumeration() == lldb::eBasicTypeNullPtr)
             && "invalid ast: target type should be a nullptr_t.");
      m_result = CreateValueNullptr(m_target, type);
      return;
    }
    case CStyleCastKind::kReference: {
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result =
          lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
      return;
    }
  }

  assert(false && "invalid ast: unexpected c-style cast kind");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CxxStaticCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  switch (node->kind()) {
    case CxxStaticCastKind::kNoOp: {
      assert(CompareTypes(type, rhs->GetCompilerType()) &&
             "invalid ast: types should be the same");
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
      return;
    }

    case CxxStaticCastKind::kArithmetic: {
      assert(IsScalar(type));
      if (IsPointerType(rhs->GetCompilerType())
          || IsNullPtrType(rhs->GetCompilerType())) {
        assert(IsBool(type) && "invalid ast: target type should be bool");
        m_result = CastPointerToBasicType(m_target, rhs, type);
      } else if (IsScalar(rhs->GetCompilerType())) {
        m_result = CastScalarToBasicType(m_target, rhs, type, m_error);
      } else if (IsEnum(rhs->GetCompilerType())) {
        m_result = CastEnumToBasicType(m_target, rhs, type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to arithmetic type");
      }
      return;
    }

    case CxxStaticCastKind::kEnumeration: {
      if (IsFloat(rhs->GetCompilerType())) {
        m_result = CastFloatToEnumType(m_target, rhs, type, m_error);
      } else if (IsInteger(rhs->GetCompilerType()) ||
                 IsEnum(rhs->GetCompilerType())) {
        m_result = CastIntegerOrEnumToEnumType(m_target, rhs, type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }

    case CxxStaticCastKind::kPointer: {
      assert(IsPointerType(type) &&
             "invalid ast: target type should be a pointer.");

      uint64_t addr = IsArrayType(rhs->GetCompilerType())
                          ? GetLoadAddress(rhs)
                          : GetUInt64(rhs);
      m_result = CreateValueFromPointer(m_target, addr, type);
      return;
    }

    case CxxStaticCastKind::kNullptr: {
      m_result = CreateValueNullptr(m_target, type);
      return;
    }

    case CxxStaticCastKind::kDerivedToBase: {
      m_result = CastDerivedToBaseType(m_target, rhs, type, node->idx());
      return;
    }

    case CxxStaticCastKind::kBaseToDerived: {
      m_result = CastBaseToDerivedType(m_target, rhs, type, node->offset());
      return;
    }
  }
}

void DILInterpreter::Visit(const CxxReinterpretCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  if (IsInteger(type)) {
    if (IsPointerType(rhs->GetCompilerType())
        || IsNullPtrType(rhs->GetCompilerType())) {
      m_result = CastPointerToBasicType(m_target, rhs, type);
    } else {
      assert(CompareTypes(type, rhs->GetCompilerType()) &&
             "invalid ast: operands should have the same type");
      // Cast value to handle type aliases.
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
    }
  } else if (IsEnum(type)) {
    assert(CompareTypes(type, rhs->GetCompilerType()) &&
           "invalid ast: operands should have the same type");
    // Cast value to handle type aliases.
    lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
    m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
  } else if (IsPointerType(type)) {
    assert((IsInteger(rhs->GetCompilerType()) ||
            IsEnum(rhs->GetCompilerType()) ||
            IsPointerType(rhs->GetCompilerType()) ||
            IsArrayType(rhs->GetCompilerType())) &&
           "invalid ast: unexpected operand to reinterpret_cast");
    uint64_t addr = IsArrayType(rhs->GetCompilerType())
                        ? GetLoadAddress(rhs)
                        : GetUInt64(rhs);
    m_result = CreateValueFromPointer(m_target, addr, type);
  } else if (IsReferenceType(type)) {
    lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
    m_result =
        lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
  } else {
    assert(false && "invalid ast: unexpected reinterpret_cast kind");
    m_result = lldb::ValueObjectSP();
  }
}

void DILInterpreter::Visit(const MemberOfNode* node) {
  assert(!node->member_index().empty() && "invalid ast: member index is empty");

  // TODO(werat): Implement address-of elision for member-of:
  //
  //  &(*ptr).foo -> (ptr + foo_offset)
  //  &ptr->foo -> (ptr + foo_offset)
  //
  // This requires calculating the offset of "foo" and generally possible only
  // for members from non-virtual bases.

  lldb::ValueObjectSP lhs = DILEvalNode(node->lhs());
  if (!lhs) {
    return;
  }

  m_result = EvaluateMemberOf(lhs, node->member_index(), node->is_synthetic());
}

void DILInterpreter::Visit(const ArraySubscriptNode* node) {
  auto base = DILEvalNode(node->base());
  if (!base) {
    return;
  }
  auto index = DILEvalNode(node->index());
  if (!index) {
    return;
  }

  // Check to see if 'base' has a synthetic value; if so, try using that.
  if (base->HasSyntheticValue()) {
    lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
    if (synthetic && synthetic != base) {
      uint64_t child_idx = GetUInt64(index);
      if (static_cast<uint32_t>(child_idx) < synthetic->GetNumChildren()) {
        lldb::ValueObjectSP child_valobj_sp =
            synthetic->GetChildAtIndex(child_idx);
        if (child_valobj_sp) {
          m_result = child_valobj_sp;
          return;
        }
      }
    }
  }

  assert(IsPointerType(base->GetCompilerType())
         && "array subscript: base must be a pointer");
  assert(IsIntegerOrUnscopedEnum(index->GetCompilerType()) &&
         "array subscript: index must be integer or unscoped enum");

  CompilerType item_type = base->GetCompilerType().GetPointeeType();
  lldb::addr_t base_addr = GetUInt64(base);

  // Create a pointer and add the index, i.e. "base + index".
  lldb::ValueObjectSP value =
      PointerAdd(CreateValueFromPointer(m_target, base_addr,
                                        item_type.GetPointerType()),
                 GetUInt64(index));

  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    m_result = value_sp;
  } else {
    Status error;
    m_result = value_sp->Dereference(error);
  }
}

void DILInterpreter::Visit(const BinaryOpNode* node) {
  // Short-circuit logical operators.
  if (node->kind() == BinaryOpKind::LAnd || node->kind() == BinaryOpKind::LOr) {
    auto lhs = DILEvalNode(node->lhs());
    if (!lhs) {
      return;
    }
    assert(IsContextuallyConvertibleToBool(lhs->GetCompilerType()) &&
           "invalid ast: must be convertible to bool");

    // For "&&" break if LHS is "false", for "||" if LHS is "true".
    bool lhs_val = GetBool(lhs);
    bool break_early =
        (node->kind() == BinaryOpKind::LAnd) ? !lhs_val : lhs_val;

    if (break_early) {
      m_result = CreateValueFromBool(m_target, lhs_val);
      return;
    }

    // Breaking early didn't happen, evaluate the RHS and use it as a result.
    auto rhs = DILEvalNode(node->rhs());
    if (!rhs) {
      return;
    }
    assert(IsContextuallyConvertibleToBool(rhs->GetCompilerType()) &&
           "invalid ast: must be convertible to bool");

    m_result = CreateValueFromBool(m_target, GetBool(rhs));
    return;
  }

  // All other binary operations require evaluating both operands.
  auto lhs = DILEvalNode(node->lhs());
  if (!lhs) {
    return;
  }
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  switch (node->kind()) {
    case BinaryOpKind::Add:
      m_result = EvaluateBinaryAddition(lhs, rhs);
      return;
    case BinaryOpKind::Sub:
      // The result type of subtraction is required because it holds the
      // correct "ptrdiff_t" type in the case of subtracting two pointers.
      m_result = EvaluateBinarySubtraction(lhs, rhs, node->result_type_deref());
      return;
    case BinaryOpKind::Mul:
      m_result = EvaluateBinaryMultiplication(lhs, rhs);
      return;
    case BinaryOpKind::Div:
      m_result = EvaluateBinaryDivision(lhs, rhs);
      return;
    case BinaryOpKind::Rem:
      m_result = EvaluateBinaryRemainder(lhs, rhs);
      return;
    case BinaryOpKind::And:
    case BinaryOpKind::Or:
    case BinaryOpKind::Xor:
      m_result = EvaluateBinaryBitwise(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::Shl:
    case BinaryOpKind::Shr:
      m_result = EvaluateBinaryShift(node->kind(), lhs, rhs);
      return;

    // Comparison operations.
    case BinaryOpKind::EQ:
    case BinaryOpKind::NE:
    case BinaryOpKind::LT:
    case BinaryOpKind::LE:
    case BinaryOpKind::GT:
    case BinaryOpKind::GE:
      m_result = EvaluateComparison(node->kind(), lhs, rhs);
      return;

    case BinaryOpKind::Assign:
      m_result = EvaluateAssignment(lhs, rhs);
      return;

    case BinaryOpKind::AddAssign:
      m_result = EvaluateBinaryAddAssign(lhs, rhs);
      return;
    case BinaryOpKind::SubAssign:
      m_result = EvaluateBinarySubAssign(lhs, rhs);
      return;
    case BinaryOpKind::MulAssign:
      m_result = EvaluateBinaryMulAssign(lhs, rhs);
      return;
    case BinaryOpKind::DivAssign:
      m_result = EvaluateBinaryDivAssign(lhs, rhs);
      return;
    case BinaryOpKind::RemAssign:
      m_result = EvaluateBinaryRemAssign(lhs, rhs);
      return;

    case BinaryOpKind::AndAssign:
    case BinaryOpKind::OrAssign:
    case BinaryOpKind::XorAssign:
      m_result = EvaluateBinaryBitwiseAssign(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::ShlAssign:
    case BinaryOpKind::ShrAssign:
      m_result = EvaluateBinaryShiftAssign(node->kind(), lhs, rhs,
                                          node->comp_assign_type());
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const UnaryOpNode* node) {
  FlowAnalysis rhs_flow(
      /* address_of_is_pending */ node->kind() == UnaryOpKind::AddrOf);

  auto rhs = DILEvalNode(node->rhs(), &rhs_flow);
  if (!rhs) {
    return;
  }

  switch (node->kind()) {
    case UnaryOpKind::Deref:
      if (rhs->GetVariable()) {
        lldb::ValueObjectSP dynamic_rhs =
            rhs->GetDynamicValue(m_default_dynamic);
        if (dynamic_rhs)
          rhs = dynamic_rhs;
      }
      m_result = EvaluateDereference(rhs);
      return;
    case UnaryOpKind::AddrOf:
      // If the address-of operation wasn't cancelled during the evaluation of
      // RHS (e.g. because of the address-of-a-dereference elision), apply it
      // here.
      if (rhs_flow.AddressOfIsPending()) {
        Status error;
        lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
        m_result = rhs_sp->AddressOf(error);
      } else {
        m_result = rhs;
      }
      return;
    case UnaryOpKind::Plus:
      m_result = rhs;
      return;
    case UnaryOpKind::Minus:
      m_result = EvaluateUnaryMinus(rhs);
      return;
    case UnaryOpKind::LNot:
      m_result = EvaluateUnaryNegation(rhs);
      return;
    case UnaryOpKind::Not:
      m_result = EvaluateUnaryBitwiseNot(rhs);
      return;
    case UnaryOpKind::PreInc:
      m_result = EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PreDec:
      m_result = EvaluateUnaryPrefixDecrement(rhs);
      return;
    case UnaryOpKind::PostInc:
      // In postfix inc/dec the result is the original value.
      m_result = Clone(rhs);
      EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PostDec:
      // In postfix inc/dec the result is the original value.
      m_result = Clone(rhs);
      EvaluateUnaryPrefixDecrement(rhs);
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const TernaryOpNode* node) {
  auto cond = DILEvalNode(node->cond());
  if (!cond) {
    return;
  }
  assert(IsContextuallyConvertibleToBool(cond->GetCompilerType()) &&
         "invalid ast: must be convertible to bool");

  // Pass down the flow analysis because the conditional operator is a "flow
  // control" construct -- LHS/RHS might be lvalues and eligible for some
  // optimizations (e.g. "&*" elision).
  if (GetBool(cond)) {
    m_result = DILEvalNode(node->lhs(), flow_analysis());
  } else {
    m_result = DILEvalNode(node->rhs(), flow_analysis());
  }
}

void DILInterpreter::Visit(const SmartPtrToPtrDecay* node) {
  auto ptr = DILEvalNode(node->ptr());
  if (!ptr) {
    return;
  }

  assert(IsSmartPtrType(ptr->GetCompilerType()) &&
         "invalid ast: must be a smart pointer");

  // Prefer synthetic value because we need LLDB machinery to "dereference" the
  // pointer for us. This is usually the default, but if the value was obtained
  // as a field of some other object, it will inherit the value from parent.
  lldb::ValueObjectSP ptr_value = ptr;
  bool prefer_synthetic_value = true;
  lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues;
  lldb::TargetSP target_sp  = ptr_value->GetTargetSP();
  if (target_sp)
    use_dynamic = target_sp->GetPreferDynamicValue();
  lldb::ValueObjectSP value_sp(DILGetSPWithLock(ptr_value, use_dynamic,
                                                prefer_synthetic_value));
  ptr_value = value_sp->GetChildAtIndex(0);

  lldb::addr_t base_addr = ptr_value->GetValueAsUnsigned(0);
  CompilerType pointer_type = ptr_value->GetCompilerType();

  m_result = CreateValueFromPointer(m_target, base_addr, pointer_type);
}

lldb::ValueObjectSP DILInterpreter::EvaluateComparison(BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  // Evaluate arithmetic operation for two integral values.
  if (IsInteger(lhs->GetCompilerType()) && IsInteger(rhs->GetCompilerType())) {
    bool ret = Compare(kind, GetInteger(lhs), GetInteger(rhs));
    return CreateValueFromBool(m_target, ret);
  }

  // Evaluate arithmetic operation for two floating point values.
  if (IsFloat(lhs->GetCompilerType()) && IsFloat(rhs->GetCompilerType())) {
    bool ret = Compare(kind, GetFloat(lhs), GetFloat(rhs));
    return CreateValueFromBool(m_target, ret);
  }

  // Evaluate arithmetic operation for two scoped enum values.
  if (IsScopedEnum(lhs->GetCompilerType()) && IsScopedEnum(rhs->GetCompilerType())) {
    bool ret = Compare(kind, GetInteger(lhs), GetInteger(rhs));
    return CreateValueFromBool(m_target, ret);
  }

  // Must be pointer/integer and/or nullptr comparison.
  size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize() * 8;

  bool ret =
      Compare(kind, llvm::APSInt(GetInteger(lhs).sextOrTrunc(ptr_size), true),
              llvm::APSInt(GetInteger(rhs).sextOrTrunc(ptr_size), true));
  return CreateValueFromBool(m_target, ret);
}

lldb::ValueObjectSP DILInterpreter::EvaluateDereference(lldb::ValueObjectSP rhs) {
  assert(IsPointerType(rhs->GetCompilerType())
         && "invalid ast: must be a pointer type");

  CompilerType pointer_type = rhs->GetCompilerType();
  lldb::addr_t base_addr = GetUInt64(rhs);

  lldb::ValueObjectSP value = CreateValueFromPointer(m_target, base_addr, pointer_type);

  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    return value_sp;
  }

  Status error;
  return value_sp->Dereference(error);
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryMinus(lldb::ValueObjectSP rhs)
{
  assert((IsInteger(rhs->GetCompilerType()) || IsFloat(rhs->GetCompilerType()))
         && "invalid ast: must be an arithmetic type");

  if (IsInteger(rhs->GetCompilerType())) {
    llvm::APSInt v = GetInteger(rhs);
    v.negate();
    return CreateValueFromAPInt(m_target, v, rhs->GetCompilerType());
  }
  if (IsFloat(rhs->GetCompilerType())) {
    llvm::APFloat v = GetFloat(rhs);
    v.changeSign();
    return CreateValueFromAPFloat(m_target, v, rhs->GetCompilerType());
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryNegation(lldb::ValueObjectSP rhs) {
  assert(IsContextuallyConvertibleToBool(rhs->GetCompilerType()) &&
         "invalid ast: must be convertible to bool");
  return CreateValueFromBool(m_target, !GetBool(rhs));
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryBitwiseNot(
    lldb::ValueObjectSP rhs) {
  assert(IsInteger(rhs->GetCompilerType()) && "invalid ast: must be an integer");
  llvm::APSInt v = GetInteger(rhs);
  v.flipAllBits();
  return CreateValueFromAPInt(m_target, v, rhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixIncrement(
    lldb::ValueObjectSP rhs)
{
  assert((IsInteger(rhs->GetCompilerType()) || IsFloat(rhs->GetCompilerType())
          || IsPointerType(rhs->GetCompilerType())) &&
         "invalid ast: must be either arithmetic type or pointer");

  if (IsInteger(rhs->GetCompilerType())) {
    llvm::APSInt v = GetInteger(rhs);
    ++v;  // Do the increment.

    Update(rhs, v);
    return rhs;
  }
  if (IsFloat(rhs->GetCompilerType())) {
    llvm::APFloat v = GetFloat(rhs);
    // Do the increment.
    v = v + llvm::APFloat(v.getSemantics(), 1ULL);

    Update(rhs, v.bitcastToAPInt());
    return rhs;
  }
  if (IsPointerType(rhs->GetCompilerType())) {
    uint64_t v = GetUInt64(rhs);
    v += GetByteSize(rhs->GetCompilerType().GetPointeeType(),
                     rhs->GetTargetSP());  // Do the increment.

    Update(rhs, llvm::APInt(64, v));
    return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixDecrement(
    lldb::ValueObjectSP rhs) {
  assert((IsInteger(rhs->GetCompilerType()) ||
          IsFloat(rhs->GetCompilerType()) ||
          IsPointerType(rhs->GetCompilerType())) &&
         "invalid ast: must be either arithmetic type or pointer");

  if (IsInteger(rhs->GetCompilerType())) {
    llvm::APSInt v = GetInteger(rhs);
    --v;  // Do the decrement.

    Update(rhs, v);
    return rhs;
  }
  if (IsFloat(rhs->GetCompilerType())) {
    llvm::APFloat v = GetFloat(rhs);
    // Do the decrement.
    v = v - llvm::APFloat(v.getSemantics(), 1ULL);

    Update(rhs, v.bitcastToAPInt());
    return rhs;
  }
  if (IsPointerType(rhs->GetCompilerType())) {
    uint64_t v = GetUInt64(rhs);
    v -= GetByteSize(rhs->GetCompilerType().GetPointeeType(), rhs->GetTargetSP());  // Do the decrement.

    Update(rhs, llvm::APInt(64, v));
    return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddition(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  // Addition of two arithmetic types.
  if (IsScalar(lhs->GetCompilerType()) && IsScalar(rhs->GetCompilerType())) {
    assert(CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Add, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }

  // Here one of the operands must be a pointer and the other one an integer.
  lldb::ValueObjectSP ptr, offset;
  if (IsPointerType(lhs->GetCompilerType())) {
    ptr = lhs;
    offset = rhs;
  } else {
    ptr = rhs;
    offset = lhs;
  }
  assert(IsPointerType(ptr->GetCompilerType()) &&
         "invalid ast: ptr must be a pointer");
  assert(IsInteger(offset->GetCompilerType()) &&
         "invalid ast: offset must be an integer");

  if (GetUInt64(ptr) == 0 && GetUInt64(offset) != 0) {
    // Binary addition with null pointer causes mismatches between LLDB and
    // lldb-eval if the offset different than zero.
    SetUbStatus(m_error, ErrorCode::kUBNullPtrArithmetic);
  }

  return PointerAdd(ptr, GetUInt64(offset));
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubtraction(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs, CompilerType result_type) {
  if (IsScalar(lhs->GetCompilerType()) && IsScalar(rhs->GetCompilerType())) {
    assert(CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Sub, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }
  assert(IsPointerType(lhs->GetCompilerType()) && "invalid ast: lhs must be a pointer");

  // "pointer - integer" operation.
  if (IsInteger(rhs->GetCompilerType())) {
    return PointerAdd(lhs, -GetUInt64(rhs));
  }

  // "pointer - pointer" operation.
  assert(IsPointerType(rhs->GetCompilerType())
         && "invalid ast: rhs must an integer or a pointer");
  assert((GetByteSize(lhs->GetCompilerType().GetPointeeType(), lhs->GetTargetSP()) ==
          GetByteSize(rhs->GetCompilerType().GetPointeeType(), rhs->GetTargetSP())) &&
         "invalid ast: pointees should be the same size");

  // Since pointers have compatible types, both have the same pointee size.
  uint64_t item_size = GetByteSize(lhs->GetCompilerType().GetPointeeType(),
                                   lhs->GetTargetSP());
  // Pointer difference is a signed value.
  int64_t diff = static_cast<int64_t>(GetUInt64(lhs) - GetUInt64(rhs));

  if (diff % item_size != 0 && diff < 0) {
    // If address difference isn't divisible by pointee size then performing
    // the operation is undefined behaviour. Note: mismatches were encountered
    // only for negative difference (diff < 0).
    SetUbStatus(m_error, ErrorCode::kUBInvalidPtrDiff);
  }

  diff /= static_cast<int64_t>(item_size);

  // Pointer difference is ptrdiff_t.
  return CreateValueFromBytes(m_target, &diff, result_type);
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMultiplication(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((IsScalar(lhs->GetCompilerType()) &&
          CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Mul, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivision(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((IsScalar(lhs->GetCompilerType()) &&
          CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  // Check for zero only for integer division.
  if (IsInteger(rhs->GetCompilerType()) && GetUInt64(rhs) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: division by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (IsInteger(rhs->GetCompilerType()) && IsInvalidDivisionByMinusOne(lhs, rhs)) {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Div, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemainder(lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((IsInteger(lhs->GetCompilerType()) && CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must be integers and have the same type");

  if (GetUInt64(rhs) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: remainder by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (IsInvalidDivisionByMinusOne(lhs, rhs)) {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOpInteger(m_target, BinaryOpKind::Rem, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwise(BinaryOpKind kind, lldb::ValueObjectSP lhs,
                                         lldb::ValueObjectSP rhs) {
  assert((IsInteger(lhs->GetCompilerType()) && CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType())) &&
         "invalid ast: operands must be integers and have the same type");
  assert((kind == BinaryOpKind::And || kind == BinaryOpKind::Or ||
          kind == BinaryOpKind::Xor) &&
         "invalid ast: operation must be '&', '|' or '^'");

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShift(BinaryOpKind kind, lldb::ValueObjectSP lhs,
                                       lldb::ValueObjectSP rhs) {
  assert(IsInteger(lhs->GetCompilerType()) &&
         IsInteger(rhs->GetCompilerType()) &&
         "invalid ast: operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr) &&
         "invalid ast: operation must be '<<' or '>>'");

  // Performing shift operation is undefined behaviour if the right operand
  // isn't in interval [0, bit-size of the left operand).
  if (GetInteger(rhs).isNegative() ||
      GetUInt64(rhs) >= GetByteSize(lhs->GetCompilerType(), lhs->GetTargetSP()) *
      CHAR_BIT) {
    SetUbStatus(m_error, ErrorCode::kUBInvalidShift);
  }

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateAssignment(lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  assert(CompareTypes(lhs->GetCompilerType(), rhs->GetCompilerType()) &&
         "invalid ast: operands must have the same type");

  Update(lhs, rhs);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs) {
  lldb::ValueObjectSP ret;

  if (IsPointerType(lhs->GetCompilerType())) {
    assert(IsInteger(rhs->GetCompilerType()) &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinaryAddition(lhs, rhs);
  } else {
    assert(IsScalar(lhs->GetCompilerType())
           && "invalid ast: lhs must be an arithmetic type");
    assert(IsBasicType(rhs->GetCompilerType()) &&
           "invalid ast: rhs must be a basic type");
    ret = CastScalarToBasicType(m_target, lhs, rhs->GetCompilerType(), m_error);
    ret = EvaluateBinaryAddition(ret, rhs);
    ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);
  }

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  lldb::ValueObjectSP ret;

  if (IsPointerType(lhs->GetCompilerType())) {
    assert(IsInteger(rhs->GetCompilerType()) &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinarySubtraction(lhs, rhs, lhs->GetCompilerType());
  } else {
    assert(IsScalar(lhs->GetCompilerType())
           && "invalid ast: lhs must be an arithmetic type");
    assert(IsBasicType(rhs->GetCompilerType()) &&
           "invalid ast: rhs must be a basic type");
    ret = CastScalarToBasicType(m_target, lhs, rhs->GetCompilerType(), m_error);
    ret = EvaluateBinarySubtraction(ret, rhs, ret->GetCompilerType());
    ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);
  }

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMulAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(IsScalar(lhs->GetCompilerType())
         && "invalid ast: lhs must be an arithmetic type");
  assert(IsBasicType(rhs->GetCompilerType())
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = CastScalarToBasicType(m_target, lhs, rhs->GetCompilerType(), m_error);
  ret = EvaluateBinaryMultiplication(ret, rhs);
  ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(IsScalar(lhs->GetCompilerType())
         && "invalid ast: lhs must be an arithmetic type");
  assert(IsBasicType(rhs->GetCompilerType())
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = CastScalarToBasicType(m_target, lhs,
                                                  rhs->GetCompilerType(),
                                                  m_error);
  ret = EvaluateBinaryDivision(ret, rhs);
  ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(IsScalar(lhs->GetCompilerType())
         && "invalid ast: lhs must be an arithmetic type");
  assert(IsBasicType(rhs->GetCompilerType())
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = CastScalarToBasicType(m_target, lhs,
                                                  rhs->GetCompilerType(),
                                                  m_error);
  ret = EvaluateBinaryRemainder(ret, rhs);
  ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwiseAssign(
    BinaryOpKind kind, lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs)
{
  switch (kind) {
    case BinaryOpKind::AndAssign:
      kind = BinaryOpKind::And;
      break;
    case BinaryOpKind::OrAssign:
      kind = BinaryOpKind::Or;
      break;
    case BinaryOpKind::XorAssign:
      kind = BinaryOpKind::Xor;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '&=', '|=' or '^='");
      break;
  }
  assert(IsScalar(lhs->GetCompilerType())
         && "invalid ast: lhs must be an arithmetic type");
  assert(IsBasicType(rhs->GetCompilerType()) &&
         "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = CastScalarToBasicType(m_target, lhs,
                                                  rhs->GetCompilerType(),
                                                  m_error);
  ret = EvaluateBinaryBitwise(kind, ret, rhs);
  ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShiftAssign(
    BinaryOpKind kind,
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs,
    CompilerType comp_assign_type)
{
  switch (kind) {
    case BinaryOpKind::ShlAssign:
      kind = BinaryOpKind::Shl;
      break;
    case BinaryOpKind::ShrAssign:
      kind = BinaryOpKind::Shr;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '<<=' or '>>='");
      break;
  }
  assert(IsScalar(lhs->GetCompilerType())
         && "invalid ast: lhs must be an arithmetic type");
  assert(IsBasicType(rhs->GetCompilerType()) &&
         "invalid ast: rhs must be a basic type");
  assert(IsInteger(comp_assign_type) &&
         "invalid ast: comp_assign_type must be an integer");

  lldb::ValueObjectSP ret = CastScalarToBasicType(m_target, lhs,
                                                  comp_assign_type, m_error);
  ret = EvaluateBinaryShift(kind, ret, rhs);
  ret = CastScalarToBasicType(m_target, ret, lhs->GetCompilerType(), m_error);

  Update(lhs, ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::PointerAdd(lldb::ValueObjectSP lhs,
                                               int64_t offset) {
  uintptr_t addr =
      GetUInt64(lhs) + offset * GetByteSize(lhs->GetCompilerType().GetPointeeType(),
                                            lhs->GetTargetSP());

  return CreateValueFromPointer(m_target, addr, lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::ResolveContextVar(
    const std::string& name) const
{
  auto it = m_context_vars.find(name);
  return it != m_context_vars.end() ? it->second : lldb::ValueObjectSP();
}

}  // namespace lldb_private
