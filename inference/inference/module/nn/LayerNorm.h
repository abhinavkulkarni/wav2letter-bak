/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <torch/csrc/api/include/torch/nn/modules/normalization.h>
#include <cstdio>
#include <memory>

#include "inference/common/IOBuffer.h"
#include "inference/module/InferenceModule.h"
#include "inference/module/ModuleParameter.h"

namespace w2l {
namespace streaming {

class LayerNorm : public InferenceModule {
 public:
  LayerNorm(int32_t featureSize, float alpha, float beta);

  virtual ~LayerNorm() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

  std::pair<InferenceModuleInfo, torch::nn::AnyModule> getTorchModule()
      const override;

 protected:
  int32_t featureSize_;
  float alpha_;
  float beta_;

 private:
  friend class cereal::access;

  LayerNorm();

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this), featureSize_, alpha_, beta_);
  }
};

struct GroupNormImpl : torch::nn::GroupNormImpl {
 private:
  float alpha, beta;

 public:
  explicit GroupNormImpl(
      int numGroups,
      int numChannels,
      float alpha,
      float beta);

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GroupNorm);

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::LayerNorm);
