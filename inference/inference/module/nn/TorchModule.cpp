//
// Created by abhinav on 10/5/20.
//

#include "inference/module/nn/TorchModule.h"
#include "LayerNorm.h"
#include "Util.h"

#include <cassert>

namespace w2l {
namespace streaming {
TorchModule::TorchModule(StackSequential module, InferenceModuleInfo info)
    : module(module), info(info) {}

TorchModule::TorchModule() {}

std::shared_ptr<ModuleProcessingState> TorchModule::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> TorchModule::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  int nFrames = inputBuf->size<float>() / info.inChannels;
  if (nFrames == 0) {
    return output;
  }
  assert(output->buffers().size() == 1);
  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  auto x = torch::from_blob(
      input->buffer(0)->data<float>(), nFrames * info.inChannels);
  x = module->forward(x).contiguous();

  auto outSize = nFrames * info.outChannels;
  outputBuf->ensure<float>(outSize);
  auto* outPtr = outputBuf->tail<float>();
  std::copy_n(x.data_ptr<float>(), outSize, outPtr);

  outputBuf->move<float>(outSize);

  inputBuf->consume<float>(nFrames * info.inChannels);
  return output;
}

std::pair<InferenceModuleInfo, torch::nn::AnyModule>
TorchModule::getTorchModule() const {
  return std::make_pair(info, torch::nn::AnyModule(module));
}

std::string TorchModule::debugString() const {
  std::stringstream ss;
  ss << module;
  ss << std::endl;
  return ss.str();
}

rapidjson::Document TorchModule::getJSON(
    rapidjson::MemoryPoolAllocator<>& allocator) const {
  return rapidjson::Document();
}
} // namespace streaming
} // namespace w2l