/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Identity.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Identity::Identity() {}

std::shared_ptr<ModuleProcessingState> Identity::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> Identity::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  assert(!input->buffers().empty());
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);
  assert(!output->buffers().empty());
  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);
  auto inLen = inputBuf->size<char>();
  outputBuf->write(inputBuf->data<char>(), inLen);
  inputBuf->consume<char>(inLen);
  return output;
}

std::string Identity::debugString() const {
  std::stringstream ss;
  ss << "Identity: {}";
  return ss.str();
}

std::shared_ptr<InferenceModuleTorchHolder> Identity::getTorchModule() const {
  auto holder = std::make_shared<InferenceModuleTorchHolder>("Identity");
  holder->anyModule = torch::nn::AnyModule(torch::nn::Identity());
  return holder;
}

rapidjson::Document Identity::getJSON(
    rapidjson::MemoryPoolAllocator<>& allocator) const {
  rapidjson::Document d(rapidjson::kObjectType);
  d.AddMember("name", "Identity", allocator);
  return d;
}

} // namespace streaming
} // namespace w2l
