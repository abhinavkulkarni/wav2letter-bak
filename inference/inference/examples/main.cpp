//
// Created by abhinav on 9/10/20.
//

#include <gflags/gflags.h>
#include <inference/module/ModuleParameter.h>
#include <inference/module/nn/LayerNorm.h>
#include <inference/module/nn/Relu.h>
#include <inference/module/nn/Residual.h>
#include <inference/module/nn/Sequential.h>
#include <inference/module/nn/TDSBlock.h>
#include <inference/module/nn/TorchUtil.h>
#include <torch/csrc/api/include/torch/all.h>
#include "Util.h"

using namespace w2l;
using namespace w2l::streaming;
using namespace torch;
using namespace rapidjson;

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

  for (int i = 0; i < 2; i++) {
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

int main(int argc, char* argv[]) {
  int T = 5;
  int numChannels = 6;

  // Create W2L Sequential InferenceModule
  auto dnnModule = createModule(numChannels);
  //  std::cout << dnnModule->debugString() << std::endl;

  // Save the model
  {
    std::ofstream ofstream("/tmp/acoustic_model.bin");
    cereal::BinaryOutputArchive ar(ofstream);
    ar(dnnModule);
    ofstream.close();
  }

  // Run module on sample data and print results
  process(dnnModule, numChannels, numChannels, T);

  // Get equivalent libtorch module
  auto holder = getTorchModule(dnnModule);
  auto module = holder->anyModule.get<StackSequential>();
  //  std::cout << module << std::endl;

  // Run module on sample data and print results
  {
    module->eval();
    auto x = torch::arange(T * numChannels).toType(kFloat);
    x = module->forward(x).contiguous();
    int N, C;
    x.sizes().size() > 1 ? (N = x.size(-2), C = x.size(-1))
                         : (N = T, C = numChannels);
    print(N, C, x.data_ptr<float>());
  }

  // Convert module to JSON and save to the disk
  {
    auto json = getJSON(module);
    json.AddMember("inShape", holder->inShape, json.GetAllocator());
    json.AddMember("inChannels", holder->inChannels, json.GetAllocator());
    json.AddMember("outShape", holder->outShape, json.GetAllocator());
    json.AddMember("outChannels", holder->outChannels, json.GetAllocator());
    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    json.Accept(writer);
    std::ofstream ofstream("/tmp/acoustic_model.json");
    std::string str(buffer.GetString(), buffer.GetSize());
    ofstream << str;
    ofstream.close();

    torch::save(module, "/tmp/acoustic_model.pth");
    ofstream = std::ofstream("/tmp/acoustic_model_libtorch_1.txt");
    ofstream << module;
    ofstream.close();
  }

  // Get the module back
  StackSequential sequential;
  {
    Document json;
    std::ifstream ifstream("/tmp/acoustic_model.json");
    IStreamWrapper isw(ifstream);
    json.ParseStream(isw);
    ifstream.close();
    sequential = getTorchModule(json);

    torch::load(sequential, "/tmp/acoustic_model.pth");
    std::ofstream ofstream =
        std::ofstream("/tmp/acoustic_model_libtorch_2.txt");
    ofstream << sequential;
    ofstream.close();
  }

  // Run module on sample data and print results
  {
    sequential->eval();
    auto x = torch::arange(T * numChannels).toType(kFloat);
    x = sequential->forward(x).contiguous();
    int N, C;
    x.sizes().size() > 1 ? (N = x.size(-2), C = x.size(-1))
                         : (N = T, C = numChannels);
    print(N, C, x.data_ptr<float>());
  }
}