#include <torch/torch.h>
#include <iostream>

using namespace std;

#define BS		12
#define SEQ_LEN		512
#define EMBEDDING_SIZE	300
#define HIDDEN_SIZE	1024
#define STRIDE	32

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

int main(void)
{
  stried_sgemm();
  //sgemm();
//  boardcast_sgemm();
  return 0;
}
