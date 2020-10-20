/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Sequential.h"
#include "inference/module/nn/TorchUtil.h"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace w2l {
namespace streaming {
Sequential::Sequential(std::vector<std::shared_ptr<InferenceModule>> modules)
    : modules_(std::move(modules)) {}

Sequential::Sequential() {}

void Sequential::add(const std::shared_ptr<InferenceModule>& module) {
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

std::tuple<
    std::string,
    std::shared_ptr<InferenceModuleInfo>,
    std::shared_ptr<InferenceModuleInfo>,
    torch::nn::AnyModule>
Sequential::getTorchModule() const {
  StackSequential sequential;

  std::shared_ptr<InferenceModuleInfo> infoFirst, infoLast;
  std::map<std::string, int> counts;

  auto getName = [](const std::string& name,
                    std::map<std::string, int>& counts) {
    if (counts.count(name) == 0)
      counts[name] = 0;
    return name + "-" + std::to_string(counts[name]++);
  };

  bool flag2D;
  auto prevOutShape = InferenceModuleInfo::shape::SHAPE_PASSTHROUGH;
  for (auto&& w2lModule : modules_) {
    auto tuple = w2lModule->getTorchModule();
    const auto& [type, infoIn, infoOut, anyModule] = tuple;

    if (infoFirst == nullptr) {
      infoFirst = infoIn;
      infoLast = infoOut;
    } else if (
        infoOut->outShape != InferenceModuleInfo::shape::SHAPE_PASSTHROUGH)
      infoLast = infoOut;

    if (infoIn->inShape == InferenceModuleInfo::shape::SHAPE_2D) {
      if (prevOutShape == InferenceModuleInfo::shape::SHAPE_3D) {
        std::vector<long> shape = {infoIn->inChannels, -1};
        sequential->push_back(
            getName("Reshape", counts), Reshape(std::move(shape)));
        std::vector<long> permutation = {1, 0};
        sequential->push_back(
            getName("Permute", counts), Permute(std::move(permutation)));
      }
      flag2D = true;
      prevOutShape = infoOut->outShape;
    } else if (infoIn->inShape == InferenceModuleInfo::shape::SHAPE_3D) {
      if (prevOutShape == InferenceModuleInfo::shape::SHAPE_2D) {
        std::vector<long> shape = {1, -1, infoIn->inChannels};
        sequential->push_back(
            getName("Reshape", counts), Reshape(std::move(shape)));
        std::vector<long> permutation = {0, 2, 1};
        sequential->push_back(
            getName("Permute", counts), Permute(std::move(permutation)));
      }
      flag2D = false;
      prevOutShape = infoOut->outShape;
    }

    if (anyModule.ptr()->as<StackSequential>()) {
      auto seqModule = anyModule.get<StackSequential>();
      std::vector<std::string> names;
      for (auto&& item : seqModule->named_children()) {
        auto name = item.key();
        name = name.substr(0, name.find('-'));
        names.push_back(name);
      }

      int i = 0;
      for (const auto& itr : *seqModule)
        sequential->push_back(getName(names[i++], counts), itr);
    } else if (type == "GroupNorm") {
      auto groupNorm = anyModule.get<GroupNormBase>();
      if (flag2D)
        sequential->push_back(
            getName("GroupNorm2D", counts), GroupNorm2D(std::move(groupNorm)));
      else
        sequential->push_back(
            getName("GroupNorm3D", counts), GroupNorm3D(std::move(groupNorm)));
    } else
      sequential->push_back(getName(type, counts), anyModule);
  }

  auto anyModule = torch::nn::AnyModule(sequential);
  return {"Sequential", infoFirst, infoLast, anyModule};
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
