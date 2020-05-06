#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <torch/torch.h>

#include <test/cpp/common/support.h>

TEST(TensorTest, StridedSgemm) {
  auto a = torch::randn({10, 20});
  auto b = torch::randn({20, 40});
  auto c = torch::matmul(a, b);
}
