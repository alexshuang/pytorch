#include <torch/torch.h>
#include <iostream>

using namespace std;
using namespace torch::nn;
namespace F = torch::nn::functional;

#define BS				12
#define SEQ_LEN			512
#define EMBEDDING_SIZE	300
#define HIDDEN_SIZE		1024
#define STRIDE			32

static void sgemm(void)
{
  auto a = torch::randn({SEQ_LEN, EMBEDDING_SIZE}).to(torch::kCUDA);
  auto b = torch::randn({EMBEDDING_SIZE, HIDDEN_SIZE}).to(torch::kCUDA);
  auto c = torch::matmul(a, b);
  cout << c.sizes() << endl;
}

static void stried_sgemm(void)
{
  auto a = torch::randn({SEQ_LEN, EMBEDDING_SIZE + STRIDE}).to(torch::kCUDA);
  auto b = torch::randn({EMBEDDING_SIZE, HIDDEN_SIZE + STRIDE}).to(torch::kCUDA);
  a = a.narrow(-1, 0, EMBEDDING_SIZE);
  b = b.narrow(-1, 0, HIDDEN_SIZE);
  cout << "A sizes: " << a.sizes() << ", B sizes: " << b.sizes() << endl;
  cout << "A strides: " << a.strides() << ", B strides: " << b.strides() << endl;
  auto c = torch::matmul(a, b);
  cout << "C sizes: " << c.sizes() << ", C strides: " << c.strides() << endl;
}

static void boardcast_sgemm(void)
{
  auto a = torch::randn({BS, SEQ_LEN, EMBEDDING_SIZE}).to(torch::kCUDA);
  auto b = torch::randn({EMBEDDING_SIZE, HIDDEN_SIZE}).to(torch::kCUDA);
  auto c = torch::matmul(a, b);
  cout << c.sizes() << endl;
}

static void hgemm(void)
{
  auto a = torch::ones({SEQ_LEN, EMBEDDING_SIZE}, torch::kFloat16).to(torch::kCUDA);
  auto b = torch::ones({EMBEDDING_SIZE, HIDDEN_SIZE}, torch::kFloat16).to(torch::kCUDA);
  auto c = torch::matmul(a, b);
  cout << c.sizes() << endl;
}

static void simple_nn(void)
{
  auto x = torch::randn({SEQ_LEN, EMBEDDING_SIZE}).to(torch::kCUDA);
  auto l = Linear(EMBEDDING_SIZE, HIDDEN_SIZE);
  l->to({torch::kCUDA, 0});

  for (auto& p : l->parameters()) {
	auto last_dim = p.data().sizes().size() - 1;
	p.data().copy_(F::pad(p.data(), F::PadFuncOptions({0, STRIDE}).mode(torch::kConstant).value(0)));//#.narrow(last_dim, 0, t.size(last_dim));
	cout << "p size: " << p.data().sizes() << ", p stride: " << p.data().strides() << endl;
  }

/*
  auto y = l(x);
  cout << "C sizes: " << y.sizes() << ", C strides: " << y.strides() << endl;
*/
}

int main(void)
{
  simple_nn();
  //stried_sgemm();
  //sgemm();
  //boardcast_sgemm();
  return 0;
}
