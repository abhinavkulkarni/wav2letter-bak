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
        "/home/abhinav/audio/cnbc-2s.wav", std::ios::binary);
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
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file loading");
    auto [infoIn, infoOut, sequential] = loadTorchModule(
        "/data/podcaster/model/wav2letter/acoustic_model.json",
        "/data/podcaster/model/wav2letter/acoustic_model_half.pth",
        "fp16");
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torchAcousticModule =
        std::make_shared<TorchModule>(infoIn, infoOut, sequential, 57, device);
  }
  auto dnnModuleLibTorch = std::make_shared<Sequential>();
  dnnModuleLibTorch->add(featureModule2);
  dnnModuleLibTorch->add(torchAcousticModule);

  {
    TimeElapsedReporter acousticLoadingElapsed(
        "libtorch acoustic model output to file");
    std::ifstream audioFile(
        "/home/abhinav/audio/cnbc-2s.wav", std::ios::binary);
    std::ofstream ofstream("./acoustic_model_libtorch.txt", std::ios::out);
    audioStreamToWordsStream(audioFile, ofstream, dnnModuleLibTorch);
  }
}

int main(int argc, char* argv[]) {
  compare();
}