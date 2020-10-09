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

std::shared_ptr<InferenceModuleTorchHolder> Sequential::getTorchModule() const {
  StackSequential sequential;

  auto ret = std::make_shared<InferenceModuleTorchHolder>("Sequential");
  std::map<std::string, int> counts;

  auto getName = [](const std::string& name,
                    std::map<std::string, int>& counts) {
    if (counts.count(name) == 0)
      counts[name] = 0;
    return name + "-" + std::to_string(counts[name]++);
  };

  auto prevOutShape = InferenceModuleTorchHolder::shape::SHAPE_PASSTHROUGH;
  for (auto&& w2lModule : modules_) {
    auto holder = w2lModule->getTorchModule();

    if (holder->inShape == InferenceModuleTorchHolder::shape::SHAPE_2D) {
      if (prevOutShape == InferenceModuleTorchHolder::shape::SHAPE_3D) {
        std::vector<long> shape = {holder->inChannels, -1};
        sequential->push_back(
            getName("Reshape", counts), Reshape(std::move(shape)));
        std::vector<long> permutation = {1, 0};
        sequential->push_back(
            getName("Permute", counts), Permute(std::move(permutation)));
      }
      prevOutShape = holder->outShape;
    } else if (holder->inShape == InferenceModuleTorchHolder::shape::SHAPE_3D) {
      if (prevOutShape == InferenceModuleTorchHolder::shape::SHAPE_2D) {
        std::vector<long> shape = {1, -1, holder->inChannels};
        sequential->push_back(
            getName("Reshape", counts), Reshape(std::move(shape)));
        std::vector<long> permutation = {0, 2, 1};
        sequential->push_back(
            getName("Permute", counts), Permute(std::move(permutation)));
      }
      prevOutShape = holder->outShape;
    }

    if (holder->anyModule.ptr()->as<StackSequential>()) {
      auto seqModule = holder->anyModule.get<StackSequential>();
      std::vector<std::string> names;
      for (auto&& item : seqModule->named_children()) {
        auto name = item.key();
        name = name.substr(0, name.find('-'));
        names.push_back(name);
      }

      int i = 0;
      for (const auto& itr : *seqModule)
        sequential->push_back(getName(names[i++], counts), itr);
    } else {
      sequential->push_back(getName(holder->type, counts), holder->anyModule);
    }
    if (ret->inChannels == -1) {
      ret->inShape = holder->inShape;
      ret->inChannels = holder->inChannels;
    }
    if (holder->outChannels != -1) {
      ret->outShape = holder->outShape;
      ret->outChannels = holder->outChannels;
    }
  }

  ret->anyModule = torch::nn::AnyModule(sequential);
  return ret;
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
