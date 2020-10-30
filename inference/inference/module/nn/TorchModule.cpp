//
// Created by abhinav on 10/5/20.
//

#include "inference/module/nn/TorchModule.h"
#include "LayerNorm.h"
#include "TorchUtil.h"

#include <cassert>
#include <utility>

namespace w2l::streaming {
TorchModule::TorchModule(
    std::shared_ptr<InferenceModuleInfo> infoIn,
    std::shared_ptr<InferenceModuleInfo> infoOut,
    StackSequential sequential,
    torch::Device device)
    : infoIn(std::move(infoIn)),
      infoOut(std::move(infoOut)),
      sequential(std::move(sequential)),
      device(device) {
  if (this->sequential->parameters().empty())
    dtype = torch::kFloat;
  else
    dtype = this->sequential->parameters().at(0).scalar_type();

  this->sequential->to(device);
  this->sequential->eval();
}

std::shared_ptr<ModuleProcessingState> TorchModule::start(
    std::shared_ptr<ModuleProcessingState> input) {
  sequential->start();
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> TorchModule::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  //  assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  int nInFrames = inputBuf->size<float>() / infoIn->inChannels;

  assert(output->buffers().size() == 1);
  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  auto x = torch::from_blob(
      input->buffer(0)->data<float>(), nInFrames * infoIn->inChannels);
  x = x.to(dtype).to(device);
  try {
    x = sequential->forward(x).contiguous();
    x = x.to(torch::kCPU).to(torch::kFloat);
    auto outSize = x.numel();
    outputBuf->ensure<float>(outSize);
    auto* outPtr = outputBuf->tail<float>();
    std::copy_n(x.data_ptr<float>(), outSize, outPtr);
    outputBuf->move<float>(outSize);
  } catch (c10::Error& e) {
    ;
  }
  auto consumedSize = nInFrames * infoIn->inChannels;
  inputBuf->consume<float>(consumedSize);
  return output;
}

std::shared_ptr<ModuleProcessingState> TorchModule::finish(
    std::shared_ptr<ModuleProcessingState> input) {
  sequential->finish();
  return run(input);
}

std::tuple<
    std::string,
    std::shared_ptr<InferenceModuleInfo>,
    std::shared_ptr<InferenceModuleInfo>,
    torch::nn::AnyModule>
TorchModule::getTorchModule() const {
  return {"TorchModule", infoIn, infoOut, torch::nn::AnyModule(sequential)};
}

std::string TorchModule::debugString() const {
  std::stringstream ss;
  ss << sequential << std::endl;
  return ss.str();
}

rapidjson::Document TorchModule::getJSON(
    rapidjson::MemoryPoolAllocator<>& allocator) const {
  return rapidjson::Document();
}
} // namespace w2l::streaming