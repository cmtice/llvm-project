#include <cstdint>

namespace test_scope {

class Value {
 public:
  Value(int x, float y) : x_(x), y_(y) {}

  // Static members
  enum ValueEnum { A, B };
  static double static_var;

 private:
  int x_;
  float y_;
};

double Value::static_var = 3.5;

}  // namespace test_scope

int
main(int argc, char **argv)
{

  test_scope::Value var(1, 2.5f);
  test_scope::Value& var_ref = var;
  uint64_t z_ = 3;

  // "raw" representation of the Value.
  int bytes[] = {1, 0x40200000};

  auto val_enum = test_scope::Value::A;
  (void)val_enum;
  (void)test_scope::Value::static_var;

  return 0; // Set a breakpoint here
}
