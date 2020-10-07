//
// Created by abhinav on 10/3/20.
//

#pragma once

#include <inference/module/nn/Sequential.h>
#include <torch/csrc/api/include/torch/nn.h>

namespace w2l {
namespace streaming {
struct StackSequentialImpl : torch::nn::SequentialImpl {
 public:
  StackSequentialImpl();

  template <typename... Modules>
  explicit StackSequentialImpl(Modules&&... modules);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(StackSequential);

struct PermuteImpl : torch::nn::Module {
 private:
  std::vector<long> vec;
  c10::IntArrayRef permutation;

 public:
  explicit PermuteImpl(std::vector<long>&& vec);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Permute);

struct ReshapeImpl : torch::nn::Module {
 private:
  std::vector<long> vec;
  c10::IntArrayRef sizes;

 public:
  explicit ReshapeImpl(std::vector<long>&& vec);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Reshape);

std::pair<InferenceModuleInfo, StackSequential> getTorchModule(
    const std::shared_ptr<Sequential>& module);

} // namespace streaming
} // namespace w2l