//
// Created by abhinav on 10/5/20.
//

#include "inference/module/nn/TorchModule.h"
#include "LayerNorm.h"
#include "TorchUtil.h"

#include <cassert>
#include <utility>

namespace w2l {
namespace streaming {
TorchModule::TorchModule(std::shared_ptr<InferenceModuleTorchHolder> holder)
    : holder(std::move(holder)) {
  this->holder->type = "TorchModule";
}

TorchModule::TorchModule() = default;

std::shared_ptr<ModuleProcessingState> TorchModule::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> TorchModule::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  // assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  int nFrames = inputBuf->size<float>() / holder->inChannels;
  if (nFrames < 80)
  {
    return output;
  }
  assert(output->buffers().size() == 1);

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  auto x = torch::from_blob(
      input->buffer(0)->data<float>(), nFrames * holder->inChannels);
  x = holder->anyModule.forward(x).contiguous();

  auto outSize = x.numel();
  auto nOutput_ = outSize / nFrames;
  outputBuf->ensure<float>(outSize);
  auto* outPtr = outputBuf->tail<float>();
  std::copy_n(x.data_ptr<float>(), outSize, outPtr);

  outputBuf->move<float>(outSize);

  inputBuf->consume<float>(nFrames * holder->inChannels);
  return output;
}

std::shared_ptr<InferenceModuleTorchHolder> TorchModule::getTorchModule()
    const {
  return holder;
}

std::string TorchModule::debugString() const {
  std::stringstream ss;
  ss << holder->anyModule.get<StackSequential>();
  ss << std::endl;
  return ss.str();
}

rapidjson::Document TorchModule::getJSON(
    rapidjson::MemoryPoolAllocator<>& allocator) const {
  return rapidjson::Document();
}
} // namespace streaming
} // namespace w2l