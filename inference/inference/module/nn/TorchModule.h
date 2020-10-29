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
#include "inference/module/nn/TorchUtil.h"

namespace w2l::streaming {
class TorchModule : public InferenceModule {
 public:
  TorchModule(
      std::shared_ptr<InferenceModuleInfo> infoIn,
      std::shared_ptr<InferenceModuleInfo> infoOut,
      StackSequential sequential,
      int minFrames = 1,
      torch::Device device = torch::kCPU);

  ~TorchModule() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> finish(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

  std::tuple<
      std::string,
      std::shared_ptr<InferenceModuleInfo>,
      std::shared_ptr<InferenceModuleInfo>,
      torch::nn::AnyModule>
  getTorchModule() const override;

  rapidjson::Document getJSON(
      rapidjson::MemoryPoolAllocator<>& allocator) const override;

 private:
  torch::Device device;
  c10::ScalarType dtype;
  int minFrames;
  StackSequential sequential;
  std::shared_ptr<InferenceModuleInfo> infoIn, infoOut;
};
} // namespace w2l::streaming

