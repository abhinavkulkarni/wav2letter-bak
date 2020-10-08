//
// Created by abhinav on 10/5/20.
//

#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstdio>
#include <memory>

#include "inference/common/IOBuffer.h"
#include "inference/module/InferenceModule.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/nn/Util.h"

namespace w2l {
namespace streaming {
class TorchModule : public InferenceModule {
 public:
  TorchModule(StackSequential module, InferenceModuleInfo info);

  virtual ~TorchModule() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

  std::pair<InferenceModuleInfo, torch::nn::AnyModule> getTorchModule()
      const override;

  rapidjson::Document getJSON(
      rapidjson::MemoryPoolAllocator<>& allocator) const;

 private:
  StackSequential module;
  InferenceModuleInfo info;
  friend class cereal::access;

  TorchModule(); // Used by Cereal for serialization.

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this));
  }
};
} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::TorchModule);
