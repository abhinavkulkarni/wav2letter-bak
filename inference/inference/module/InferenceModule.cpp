/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/InferenceModule.h"

#include <utility>

#include "inference/common/DefaultMemoryManager.h"

namespace w2l {
namespace streaming {

InferenceModule::InferenceModule()
    : memoryManager_(std::make_shared<DefaultMemoryManager>()) {}

void InferenceModule::setMemoryManager(
    std::shared_ptr<MemoryManager> memoryManager) {
  memoryManager_ = memoryManager;
}

InferenceModuleInfo::InferenceModuleInfo(
    InferenceModuleInfo::shape inShape,
    int inChannels,
    InferenceModuleInfo::shape outShape,
    int outChannels,
    std::map<std::string, int> kwargs)
    : inShape(inShape),
      inChannels(inChannels),
      outShape(outShape),
      outChannels(outChannels),
      kwargs(std::move(kwargs)) {}
} // namespace streaming
} // namespace w2l
