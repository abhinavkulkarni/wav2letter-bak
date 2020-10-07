/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/LayerNorm.h"

#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "inference/common/Functions.h"

namespace {
constexpr const float kEpsilon = 1e-5; // min stddev to avoid division by zero
}

namespace w2l {
namespace streaming {

LayerNorm::LayerNorm() : LayerNorm(1, 1.0, 1.0) {}

LayerNorm::LayerNorm(int32_t featureSize, float alpha, float beta)
    : featureSize_(featureSize), alpha_(alpha), beta_(beta) {
  if (featureSize <= 0) {
    std::stringstream ss;
    ss << "Invalid argument at LayerNorm::LayerNorm(featureSize=" << featureSize
       << " alpha=" << alpha << " beta_=" << beta << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::shared_ptr<ModuleProcessingState> LayerNorm::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> LayerNorm::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  const int T = inputBuf->size<float>() / featureSize_;
  if (T == 0) {
    return output;
  }

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  const int N = T * featureSize_;
  outputBuf->ensure<float>(N);
  float mean = 0.0, stddev = 0.0;
  for (int t = 0; t < T; ++t) {
    const float* inPtr = inputBuf->data<float>();
    float* outPtr = outputBuf->tail<float>();
    meanAndStddev(inPtr, featureSize_, /* output */ mean, /* output */ stddev);

    if (stddev <= kEpsilon) {
      stddev = 1.0;
    }
    meanNormalize(
        inPtr, featureSize_, mean, stddev, alpha_, beta_, /* output */ outPtr);

    inputBuf->consume<float>(featureSize_);
    outputBuf->move<float>(featureSize_);
  }

  return output;
}

std::string LayerNorm::debugString() const {
  std::stringstream ss;
  ss << "LayerNorm:{featureSize=" << featureSize_ << " alpha=" << alpha_
     << " beta_=" << beta_ << "}";
  return ss.str();
}

std::pair<InferenceModuleInfo, torch::nn::AnyModule> LayerNorm::getTorchModule()
    const {
  return std::make_pair(
      InferenceModuleInfo(),
      torch::nn::AnyModule(GroupNorm(1, featureSize_, alpha_, beta_).ptr()));
}

GroupNormImpl::GroupNormImpl(
    int numGroups,
    int numChannels,
    float alpha,
    float beta)
    : torch::nn::GroupNormImpl(numGroups, numChannels),
      alpha(alpha),
      beta(beta) {
  auto &weight = torch::nn::GroupNormImpl::weight,
       &bias = torch::nn::GroupNormImpl::bias;
  weight = weight * alpha;
  bias = bias + beta;
}

torch::Tensor GroupNormImpl::forward(torch::Tensor x) {
  if (x.sizes().size() == 3)
    return torch::nn::GroupNormImpl::forward(x.permute({2, 1, 0}))
        .permute({2, 1, 0});
  else
    return torch::nn::GroupNormImpl::forward(x.unsqueeze(-1)).squeeze(-1);
}

void GroupNormImpl::pretty_print(std::ostream& stream) const {
  auto& options = torch::nn::GroupNormImpl::options;
  stream << "GroupNorm(" << options.num_groups() << ", "
         << options.num_channels() << ", alpha=" << alpha << ", beta=" << beta
         << ")";
}

} // namespace streaming
} // namespace w2l
