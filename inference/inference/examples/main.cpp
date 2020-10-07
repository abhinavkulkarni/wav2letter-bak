//
// Created by abhinav on 9/10/20.
//

#include <gflags/gflags.h>
#include <inference/module/ModuleParameter.h>
#include <inference/module/nn/LayerNorm.h>
#include <inference/module/nn/Relu.h>
#include <inference/module/nn/Sequential.h>
#include <inference/module/nn/TDSBlock.h>
#include <inference/module/nn/TorchModule.h>
#include <inference/module/nn/Util.h>
#include "Util.h"

using namespace w2l;
using namespace w2l::streaming;
using namespace torch;

std::shared_ptr<ModuleParameter> initialize_weights(int size) {
  std::vector<float> vec(size);
  for (int i = 0; i < size; i++)
    vec[i] = i * (i % 2 ? 1 : -1);
  auto weights = std::make_shared<ModuleParameter>(
      streaming::DataType::FLOAT, vec.data(), vec.size());
  return weights;
}

void print(int N, int C, const float* buf) {
  std::stringstream ss;
  ss.precision(4);
  for (int i = 0; i < N; i++) {
    ss << "T:" << std::setw(3) << i << std::setw(3) << "|";
    for (int j = 0; j < C; j++)
      ss << std::setw(10) << *(buf + i * C + j);
    ss << std::endl;
  }

  std::cout << ss.str() << std::endl;
}

void process(
    const std::shared_ptr<InferenceModule>& dnnModule,
    int inChannels,
    int outChannels,
    int T) {
  auto input = std::make_shared<streaming::ModuleProcessingState>(1);
  auto inputBuffer = input->buffer(0);

  auto output = dnnModule->start(input);
  auto outputBuffer = output->buffer(0);

  float buffer[inChannels * T];
  for (int i = 0; i < T * inChannels; i++)
    buffer[i] = i;

  inputBuffer->write(buffer, inChannels * T);
  dnnModule->run(input);
  dnnModule->finish(input);
  auto* data = outputBuffer->data<float>();
  int size = outputBuffer->size<float>();
  print(size / outChannels, outChannels, data);
  outputBuffer->consume<float>(size);
}

std::shared_ptr<Sequential> createModule(int inChannels) {
  auto dnnModule = std::make_shared<Sequential>();
  std::vector<float> input;
  int outChannels = inChannels;

  {
    int kernelSize = 3;
    auto weights = initialize_weights(inChannels * outChannels * kernelSize);
    auto bias = initialize_weights(outChannels);
    auto conv1dModule = createConv1d(
        inChannels, outChannels, kernelSize, 1, 1, 1, weights, bias);
    auto reluModule = std::make_shared<Relu>(streaming::DataType::FLOAT);
    weights = initialize_weights(inChannels * outChannels);
    auto linearModule = createLinear(inChannels, outChannels, weights, bias);
    auto sequential = std::make_shared<Sequential>();
    sequential->add(conv1dModule);
    sequential->add(reluModule);
    sequential->add(linearModule);
    dnnModule->add(sequential);
  }

  { dnnModule->add(std::make_shared<LayerNorm>(outChannels, 1, 0)); }
  { dnnModule->add(std::make_shared<Relu>(streaming::DataType::FLOAT)); }

  {
    int kernelSize = 3;
    auto weights = initialize_weights(inChannels * outChannels * kernelSize);
    auto bias = initialize_weights(outChannels);
    auto conv1dModule = createConv1d(
        inChannels, outChannels, kernelSize, 1, 1, 1, weights, bias);

    weights = initialize_weights(inChannels * outChannels);
    bias = initialize_weights(outChannels);
    auto linearModule1 = createLinear(inChannels, outChannels, weights, bias);
    auto linearModule2 = createLinear(inChannels, outChannels, weights, bias);
    auto tdsBlockModule = std::make_shared<TDSBlock>(
        conv1dModule,
        std::make_shared<LayerNorm>(outChannels, 1, 0),
        linearModule1,
        linearModule2,
        std::make_shared<LayerNorm>(outChannels, 1, 0),
        streaming::DataType::FLOAT,
        streaming::DataType::FLOAT);
    dnnModule->add(tdsBlockModule);
  }

  return dnnModule;
}

DEFINE_string(
    input_files_base_path,
    ".",
    "path is added as prefix to input files unless the input file"
    " is a full path.");
DEFINE_string(
    acoustic_module_file,
    "acoustic_model.bin",
    "binary file containing acoustic module parameters.");

std::string GetInputFileFullPath(const std::string& fileName) {
  return GetFullPath(fileName, FLAGS_input_files_base_path);
}

void fbgemmTolibtorch(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::shared_ptr<streaming::Sequential> acousticModule;
  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file loading");
    std::ifstream amFile(
        GetInputFileFullPath(FLAGS_acoustic_module_file), std::ios::binary);
    if (!amFile.is_open()) {
      throw std::runtime_error(
          "failed to open acoustic model file=" +
          GetInputFileFullPath(FLAGS_acoustic_module_file) + " for reading");
    }
    cereal::BinaryInputArchive ar(amFile);
    ar(acousticModule);
  }

  torch::NoGradGuard no_grad;
  auto pair = getTorchModule(acousticModule);

  auto info = pair.first;
  auto module = pair.second;

  module->eval();
  std::cout << module << std::endl;

  int T = 100;
  process(acousticModule, info.inChannels, info.outChannels, T);

  auto x = torch::arange(T * info.inChannels).toType(torch::kFloat);
  x = module->forward(x).contiguous();
  auto N = x.size(-2), C = x.size(-1);
  print(N, C, x.data_ptr<float>());
}

int main(int argc, char* argv[]) {
  int T = 5;
  int numChannels = 6;

  // Create W2L Sequential InferenceModule
  auto dnnModule = createModule(numChannels);
  std::cout << dnnModule->debugString() << std::endl;

  // Run module on a sample data
  process(dnnModule, numChannels, numChannels, T);

  // Get equivalent libtorch module
  auto pair = getTorchModule(dnnModule);
  auto info = pair.first;
  auto module = pair.second;
  std::cout << module << std::endl;

  // Run module on a sample data
  auto x = torch::arange(T * numChannels).toType(kFloat);
  x = module->forward(x).contiguous();
  auto N = x.size(-2), C = x.size(-1);
  print(N, C, x.data_ptr<float>());

  // Get equivalent W2L torch InferenceModule
  auto torchModule = std::make_shared<TorchModule>(module, info);
  std::cout << torchModule->debugString() << std::endl;

  // Run module on a sample data
  process(torchModule, numChannels, numChannels, T);

  // Convert the accoustic module into a libtorch module
  fbgemmTolibtorch(argc, argv);
}