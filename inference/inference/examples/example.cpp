//
// Created by abhinav on 10/17/20.
//

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <gflags/gflags.h>
#include <torch/csrc/api/include/torch/serialize.h>

#include <torch/torch.h>
#include "inference/decoder/Decoder.h"
#include "inference/examples/AudioToWords.h"
#include "inference/examples/Util.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

using namespace w2l;
using namespace w2l::streaming;

std::shared_ptr<ModuleParameter> getWeights(int size) {
  auto x = torch::randn(size);
  x = x * 0;
  auto* buf = x.data_ptr<float>();
  auto weights = std::make_shared<ModuleParameter>(DataType::FLOAT, buf, size);
  return weights;
}

std::shared_ptr<Sequential> getAcousticModule() {
  auto dnnModule = std::make_shared<Sequential>();

  int outChannels = 4;

  auto weights = getWeights(4 * outChannels * 9),
       bias = getWeights(outChannels);
  auto conv1dModule =
      createConv1d(4, outChannels, 9, 1, {7, 1}, 1, weights, bias);

  auto residualModule =
      std::make_shared<Residual>(conv1dModule, DataType::FLOAT);
  dnnModule->add(residualModule);

  auto [infoIn, infoOut, sequential] = getTorchModule(dnnModule);

  auto input = std::make_shared<ModuleProcessingState>(1);
  auto output = dnnModule->start(input);

  int T = 2;
  for (int iter = 0; iter < 3; iter++) {
    auto start = iter * T * 4, end = (iter + 1) * T * 4;
    auto x = torch::arange(start, end).toType(torch::kFloat);
    std::cout << "Input=\t" << x.reshape({1, -1}) << std::endl;
    input->buffer(0)->write(x.data_ptr<float>(), x.numel());
    output = dnnModule->run(input);
    auto nOutFrames = output->buffer(0)->size<float>() / 4;
    auto y = torch::from_blob(output->buffer(0)->data<float>(), nOutFrames * 4);
    output->buffer(0)->consume<float>(nOutFrames * 4);
    std::cout << "Output=\t" << y.reshape({1, -1}) << std::endl;
    x = sequential->forward(x).contiguous();
    std::cout << "Output=\t" << x.reshape({1, -1}) << std::endl;
  }

  return dnnModule;
}

void audioStreamToWordsStream(
    std::istream& inputAudioStream,
    std::ostream& outputWordsStream,
    const std::shared_ptr<Sequential>& dnnModule) {
  constexpr const int lookBack = 0;
  constexpr const size_t kWavHeaderNumBytes = 44;
  constexpr const float kMaxUint16 = static_cast<float>(0x8000);
  constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
  constexpr const int kChunkSizeMsec = 500;

  inputAudioStream.ignore(kWavHeaderNumBytes);

  const int minChunkSize = kChunkSizeMsec * kAudioWavSamplingFrequency / 1000;
  auto input = std::make_shared<ModuleProcessingState>(1);
  auto inputBuffer = input->buffer(0);
  int audioSampleCount = 0;

  // The same output object is returned by start(), run() and finish()
  auto output = dnnModule->start(input);
  auto outputBuffer = output->buffer(0);
  bool finish = false;

  auto nTokens = 9998;

  int iter = 0;
  while (!finish) {
    iter += 1;
    int curChunkSize = readTransformStreamIntoBuffer<int16_t, float>(
        inputAudioStream, inputBuffer, minChunkSize, [](int16_t i) -> float {
          return static_cast<float>(i) / kMaxUint16;
        });

    float* data;
    int size;

    std::cout << "Iter=" << iter
              << "\tinputBuf size=" << inputBuffer->size<float>();

    if (curChunkSize >= minChunkSize) {
      dnnModule->run(input);
    } else {
      dnnModule->finish(input);
      finish = true;
    }
    data = outputBuffer->data<float>();
    size = outputBuffer->size<float>();

    // Consume and prune
    const int nFramesOut = outputBuffer->size<float>() / nTokens;
    std::cout << "\toutputBuf size=" << size << std::endl;

    for (int i = 0; i < nFramesOut * nTokens; i++)
      outputWordsStream << data[i] << std::endl;
    outputBuffer->consume<float>(nFramesOut * nTokens);
  }
};

void compare() {
  std::shared_ptr<Sequential> featureModule1;
  {
    TimeElapsedReporter feturesLoadingElapsed("features model file loading");
    std::ifstream featFile(
        "/data/podcaster/model/wav2letter/feature_extractor.bin",
        std::ios::binary);
    cereal::BinaryInputArchive ar(featFile);
    ar(featureModule1);
  }

  std::shared_ptr<Sequential> acousticModule;
  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file loading");
    std::ifstream amFile(
        "/data/podcaster/model/wav2letter/acoustic_model.bin",
        std::ios::binary);
    cereal::BinaryInputArchive ar(amFile);
    ar(acousticModule);
  }
  auto dnnModuleFbGemm = std::make_shared<Sequential>();
  dnnModuleFbGemm->add(featureModule1);
  dnnModuleFbGemm->add(acousticModule);

  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model output to file");
    std::ifstream audioFile(
        "/home/abhinav/Downloads/output-1s.wav", std::ios::binary);
    std::ofstream ofstream("./acoustic_model_fbgemm.txt", std::ios::out);
    audioStreamToWordsStream(audioFile, ofstream, dnnModuleFbGemm);
  }

  std::shared_ptr<Sequential> featureModule2;
  {
    TimeElapsedReporter feturesLoadingElapsed("features model file loading");
    std::ifstream featFile(
        "/data/podcaster/model/wav2letter/feature_extractor.bin",
        std::ios::binary);
    cereal::BinaryInputArchive ar(featFile);
    ar(featureModule2);
  }

  std::shared_ptr<TorchModule> torchAcousticModule;
  {
    TimeElapsedReporter acousticLoadingElapsed(
        "torch acoustic model file loading");
    rapidjson::Document json;
    std::ifstream amDefinitionFile(
        "/data/podcaster/model/wav2letter/acoustic_model.json", std::ios::in);
    rapidjson::IStreamWrapper isw(amDefinitionFile);
    json.ParseStream(isw);
    auto sequential = getTorchModule(json);
    std::shared_ptr<InferenceModuleInfo> infoIn, infoOut;

    torch::load(
        sequential, "/data/podcaster/model/wav2letter/acoustic_model.pth");

    for (auto& name : {"inInfo", "outInfo"}) {
      auto obj = json[name].GetObject();
      std::map<std::string, int> kwargs;
      if (obj.FindMember("kernelSize") != obj.MemberEnd())
        kwargs = {{"kernelSize", obj["kernelSize"].GetInt()}};

      auto inShape =
          static_cast<InferenceModuleInfo::shape>(obj["inShape"].GetInt());
      auto outShape =
          static_cast<InferenceModuleInfo::shape>(obj["outShape"].GetInt());
      auto inChannels = obj["inChannels"].GetInt();
      auto outChannels = obj["outChannels"].GetInt();
      auto info = std::make_shared<InferenceModuleInfo>(
          inShape, inChannels, outShape, outChannels, kwargs);
      if (strcmp(name, "inInfo") == 0)
        infoIn = info;
      else
        infoOut = info;
    }
    torchAcousticModule =
        std::make_shared<TorchModule>(infoIn, infoOut, sequential, 57);
  }
  auto dnnModuleLibTorch = std::make_shared<Sequential>();
  dnnModuleLibTorch->add(featureModule2);
  dnnModuleLibTorch->add(torchAcousticModule);

  {
    TimeElapsedReporter acousticLoadingElapsed(
        "libtorch acoustic model output to file");
    std::ifstream audioFile(
        "/home/abhinav/Downloads/output-1s.wav", std::ios::binary);
    std::ofstream ofstream("./acoustic_model_libtorch.txt", std::ios::out);
    audioStreamToWordsStream(audioFile, ofstream, dnnModuleLibTorch);
  }
}

int main(int argc, char* argv[]) {
  //  getAcousticModule();
  compare();
}