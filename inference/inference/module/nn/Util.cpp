//
// Created by abhinav on 10/3/20.
//

#include "inference/module/nn/Util.h"

namespace w2l {
namespace streaming {

torch::Tensor StackSequentialImpl::forward(torch::Tensor x) {
  return torch::nn::SequentialImpl::forward(x);
}

template <typename... Modules>
StackSequentialImpl::StackSequentialImpl(Modules&&... modules)
    : torch::nn::SequentialImpl(modules...) {}

StackSequentialImpl::StackSequentialImpl() : torch::nn::SequentialImpl() {}

void StackSequentialImpl::pretty_print(std::ostream& stream) const {
  SequentialImpl::pretty_print(stream);
}

PermuteImpl::PermuteImpl(std::vector<long>&& vec) {
  this->vec = std::move(vec);
  permutation = c10::IntArrayRef(this->vec);
}

torch::Tensor PermuteImpl::forward(torch::Tensor x) {
  x = x.permute(permutation);
  return x;
}

void PermuteImpl::pretty_print(std::ostream& stream) const {
  stream << "Permute(" << permutation << ")";
}

ReshapeImpl::ReshapeImpl(std::vector<long>&& vec) {
  this->vec = std::move(vec);
  sizes = c10::IntArrayRef(this->vec);
}

torch::Tensor ReshapeImpl::forward(torch::Tensor x) {
  x = x.reshape(sizes);
  return x;
}

void ReshapeImpl::pretty_print(std::ostream& stream) const {
  stream << "Reshape(" << sizes << ")";
}

std::pair<InferenceModuleInfo, StackSequential> getTorchModule(
    const std::shared_ptr<Sequential>& module) {
  auto pair = module->getTorchModule();
  auto info = pair.first;
  auto sequential = StackSequential();

  if (info.inShape == InferenceModuleInfo::shape::SHAPE_2D) {
    std::vector<long> shape = {-1, info.inChannels};
    sequential->push_back(Reshape(std::move(shape)));
  } else if (info.inShape == InferenceModuleInfo::shape::SHAPE_3D) {
    std::vector<long> shape = {1, -1, info.inChannels};
    sequential->push_back(Reshape(std::move(shape)));
    std::vector<long> permutation = {0, 2, 1};
    sequential->push_back(Permute(std::move(permutation)));
  }

  auto seqModule = pair.second.get<StackSequential>();
  for (auto itr = seqModule->begin(); itr != seqModule->end(); itr++)
    sequential->push_back(*itr);

  if (info.outShape == InferenceModuleInfo::shape::SHAPE_2D) {
  } else if (info.outShape == InferenceModuleInfo::shape::SHAPE_3D) {
    std::vector<long> permutation = {0, 2, 1};
    sequential->push_back(Permute(std::move(permutation)));
  }

  return std::make_pair(info, sequential);
}
} // namespace streaming
} // namespace w2l
