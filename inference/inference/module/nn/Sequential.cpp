/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Sequential.h"
#include "inference/module/nn/Util.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {
Sequential::Sequential(std::vector<std::shared_ptr<InferenceModule>> modules)
    : modules_(modules) {}

Sequential::Sequential() {}

void Sequential::add(std::shared_ptr<InferenceModule> module) {
  modules_.push_back(module);
}

std::shared_ptr<ModuleProcessingState> Sequential::start(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->start(intermediateInput);
  }
  return intermediateInput;
}

std::shared_ptr<ModuleProcessingState> Sequential::run(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->run(intermediateInput);
  }
  return intermediateInput;
}

std::shared_ptr<ModuleProcessingState> Sequential::finish(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->finish(intermediateInput);
  }
  return intermediateInput;
}

void Sequential::setMemoryManager(
    std::shared_ptr<MemoryManager> memoryManager) {
  InferenceModule::setMemoryManager(memoryManager);
  for (auto module : modules_) {
    module->setMemoryManager(memoryManager);
  }
}

std::string Sequential::debugString() const {
  std::stringstream ss;
  ss << "Sequential: { \n";
  for (auto& module : modules_) {
    ss << module->debugString() << "\n";
  }

  ss << "}";
  return ss.str();
}

std::pair<InferenceModuleInfo, torch::nn::AnyModule>
Sequential::getTorchModule() const {
  StackSequential sequential;

  InferenceModuleInfo ret;

  auto prevOutShape = InferenceModuleInfo::shape::SHAPE_PASSTHROUGH;
  for (auto&& w2lModule : modules_) {
    auto pair = w2lModule->getTorchModule();
    auto info = pair.first;
    auto module = pair.second;

    if (info.inShape == InferenceModuleInfo::shape::SHAPE_2D) {
      if (prevOutShape == InferenceModuleInfo::shape::SHAPE_3D) {
        std::vector<long> shape = {info.inChannels, -1};
        sequential->push_back(Reshape(std::move(shape)));
        std::vector<long> permutation = {1, 0};
        sequential->push_back(Permute(std::move(permutation)));
      }
      prevOutShape = info.outShape;
    } else if (info.inShape == InferenceModuleInfo::shape::SHAPE_3D) {
      if (prevOutShape == InferenceModuleInfo::shape::SHAPE_2D) {
        std::vector<long> shape = {1, -1, info.inChannels};
        sequential->push_back(Reshape(std::move(shape)));
        std::vector<long> permutation = {0, 2, 1};
        sequential->push_back(Permute(std::move(permutation)));
      }
      prevOutShape = info.outShape;
    }

    if (module.ptr()->as<StackSequential>()) {
      auto seqModule = module.get<StackSequential>();
      for (auto itr = seqModule->begin(); itr != seqModule->end(); itr++)
        sequential->push_back(*itr);
    } else
      sequential->push_back(module);
    if (ret.inChannels == -1) {
      ret.inShape = info.inShape;
      ret.inChannels = info.inChannels;
    }
    if (info.outChannels != -1) {
      ret.outShape = info.outShape;
      ret.outChannels = info.outChannels;
    }
  }

  return std::make_pair(ret, torch::nn::AnyModule(sequential.ptr()));
}

rapidjson::Document Sequential::getJSON(
    rapidjson::MemoryPoolAllocator<>& allocator) const {
  rapidjson::Document d(rapidjson::kObjectType);

  d.AddMember("name", "Sequential", allocator);
  rapidjson::Document children(rapidjson::kArrayType);
  for (const auto& module : modules_)
    children.PushBack(module->getJSON(allocator).Move(), allocator);
  d.AddMember("children", children, allocator);

  return d;
}

} // namespace streaming
} // namespace w2l
