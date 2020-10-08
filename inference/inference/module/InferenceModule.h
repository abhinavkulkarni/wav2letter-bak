/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/external/rapidjson/document.h>
#include <cereal/types/polymorphic.hpp>
#include <torch/csrc/api/include/torch/nn/modules/container/any.h>
#include <memory>
#include <vector>

#include "inference/common/MemoryManager.h"
#include "inference/module/ModuleProcessingState.h"

namespace w2l {
namespace streaming {

struct InferenceModuleInfo {
 public:
  enum shape { SHAPE_2D, SHAPE_3D, SHAPE_PASSTHROUGH };

  shape inShape, outShape;
  int inChannels, outChannels;

  InferenceModuleInfo() {
    inShape = outShape = shape::SHAPE_PASSTHROUGH;
    inChannels = outChannels = -1;
  }

  InferenceModuleInfo(
      shape inShape,
      int inChannels,
      shape outShape,
      int outChannels)
      : inShape(inShape),
        inChannels(inChannels),
        outShape(outShape),
        outChannels(outChannels) {}
};

// Base class for all modules of the inference processing graph, including:
// - Neural network layers.
// - Activation functions.
// - Composite modules.
// InferenceModule are intended to be used as building blocks to
// rapidly construct and test deep neural networks.
class InferenceModule {
 public:
  // memeoryManager is used by some modules to allocate temporary workspaces.
  InferenceModule();
  virtual ~InferenceModule() {}

  // The return value is the output state of the module. In the case of modules
  // that implement simple activation function such as ReLU, the output can
  // simply be written into the buffer space of the input. In the case of
  // modules that have more involved computation, the output has a vector
  // of buffers. The size of the vector is the size of the module's output.
  virtual std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> /*input*/) = 0;

  virtual std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) = 0;

  virtual std::shared_ptr<ModuleProcessingState> finish(
      std::shared_ptr<ModuleProcessingState> input) {
    return run(input);
  }

  virtual void clear() {}

  virtual void setMemoryManager(std::shared_ptr<MemoryManager> memoryManager);

  virtual std::string debugString() const = 0;

  virtual std::pair<InferenceModuleInfo, torch::nn::AnyModule> getTorchModule()
      const = 0;

  virtual rapidjson::Document getJSON(
      rapidjson::MemoryPoolAllocator<>& allocator) const = 0;

 protected:
  std::shared_ptr<MemoryManager> memoryManager_;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& /*ar*/) {}
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::InferenceModule);
