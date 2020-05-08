#include <torch/torch.h>
#include <iostream>

using namespace std;

#define BS		12
#define SEQ_LEN		512
#define EMBEDDING_SIZE	300
#define HIDDEN_SIZE	1024

static void sgemm(void)
{
  auto a = torch::randn({SEQ_LEN, EMBEDDING_SIZE}).to(torch::kCUDA);
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
  sgemm();
  return 0;
}
