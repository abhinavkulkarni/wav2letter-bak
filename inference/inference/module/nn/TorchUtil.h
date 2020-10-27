//
// Created by abhinav on 10/3/20.
//

#pragma once

#include <cereal/external/rapidjson/document.h>
#include <inference/module/nn/Sequential.h>
#include <torch/csrc/api/include/torch/nn.h>

namespace w2l::streaming {
struct StackSequentialImpl : torch::nn::SequentialImpl {
 public:
  StackSequentialImpl();

  template <typename... Modules>
  explicit StackSequentialImpl(Modules&&... modules);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);

  void finish();
};

TORCH_MODULE(StackSequential);

struct PermuteImpl : torch::nn::Module {
 public:
  std::vector<long> vec;
  c10::IntArrayRef permutation;

  explicit PermuteImpl(std::vector<long>&& vec);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Permute);

struct ReshapeImpl : torch::nn::Module {
 public:
  std::vector<long> vec;
  c10::IntArrayRef sizes;

  explicit ReshapeImpl(std::vector<long>&& vec);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Reshape);

struct W2LGroupNormImpl : torch::nn::Module {
 public:
  W2LGroupNormImpl(float alpha, float beta);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);

  torch::Tensor alpha, beta;
};

TORCH_MODULE(W2LGroupNorm);

struct ResidualTorchImpl : torch::nn::Module {
 public:
  std::string name;
  torch::nn::AnyModule anyModule;

  explicit ResidualTorchImpl(std::string name, torch::nn::AnyModule anyModule);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::Tensor padding;
};

TORCH_MODULE(ResidualTorch);

struct Conv1dUnequalPaddingImpl : torch::nn::Conv1dImpl {
 public:
  int leftPadding, rightPadding, groups;

  Conv1dUnequalPaddingImpl(
      int inChannels,
      int outChannels,
      int kernelSize,
      int stride,
      int leftPadding,
      int rightPadding,
      int groups);

  torch::Tensor forward(torch::Tensor x);

  void pretty_print(std::ostream& stream) const override;

  void finish();

 private:
  torch::Tensor leftPaddingTensor, rightPaddingTensor;
};

TORCH_MODULE(Conv1dUnequalPadding);

std::tuple<
    std::shared_ptr<InferenceModuleInfo>,
    std::shared_ptr<InferenceModuleInfo>,
    StackSequential>
getTorchModule(const std::shared_ptr<Sequential>& module);

rapidjson::Document getJSON(const std::shared_ptr<InferenceModule>& dnnModule);

rapidjson::Document getJSON(const StackSequential& seqModule);

StackSequential getTorchModule(const rapidjson::Document& json);
} // namespace w2l::streaming