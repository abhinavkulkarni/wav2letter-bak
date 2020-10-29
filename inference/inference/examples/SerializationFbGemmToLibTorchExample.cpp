//
// Created by abhinav on 10/10/20.
//

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <gflags/gflags.h>
#include <inference/module/nn/TorchUtil.h>
#include <torch/csrc/api/include/torch/all.h>

#include "inference/examples/AudioToWords.h"
#include "inference/examples/Util.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

using namespace w2l;
using namespace w2l::streaming;

DEFINE_string(
    input_files_base_path,
    ".",
    "path is added as prefix to input files unless the input file"
    " is a full path.");
DEFINE_string(
    acoustic_module_file,
    "acoustic_model.bin",
    "binary file containing acoustic module parameters.");
DEFINE_string(
    output_files_base_path,
    ".",
    "path is added as prefix to output files unless the output file"
    " is a full path.");
DEFINE_string(
    acoustic_module_definition_file,
    "acoustic_model.json",
    "JSON file containing libtorch acoustic module definition.");
DEFINE_string(
    acoustic_module_parameter_file,
    "acoustic_model.pth",
    "binary file containing libtorch acoustic module parameters.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::shared_ptr<streaming::Sequential> acousticModule;

  // Read acoustic binary file
  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file loading");
    std::ifstream amFile(
        GetFullPath(FLAGS_acoustic_module_file, FLAGS_input_files_base_path),
        std::ios::binary);
    if (!amFile.is_open()) {
      throw std::runtime_error(
          "failed to open acoustic model file=" +
          GetFullPath(FLAGS_acoustic_module_file, FLAGS_input_files_base_path) +
          " for reading");
    }
    cereal::BinaryInputArchive ar(amFile);
    ar(acousticModule);
    amFile.close();
  }

  // Get equivalent LibTorch module
  torch::NoGradGuard no_grad;

  StackSequential sequential;
  std::shared_ptr<InferenceModuleInfo> infoIn, infoOut;
  {
    TimeElapsedReporter acousticLoadingElapsed("FBGEMM to LibTorch conversion");
    auto tuple = getTorchModule(acousticModule);
    auto& [infoIn_, infoOut_, sequential_] = tuple;
    infoIn = infoIn_;
    infoOut = infoOut_;
    sequential = std::move(sequential_);
  }

  // Get module definition
  auto json = getJSON(sequential);
  // Save model definition and parameters
  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file saving");

    for (auto const& info : {infoIn, infoOut}) {
      rapidjson::Document obj(rapidjson::kObjectType);
      obj.AddMember("inShape", info->inShape, json.GetAllocator());
      obj.AddMember("inChannels", info->inChannels, json.GetAllocator());
      if (not info->kwargs.empty()) {
        obj.AddMember(
            "kernelSize", info->kwargs["kernelSize"], json.GetAllocator());
      }
      obj.AddMember("outShape", info->outShape, json.GetAllocator());
      obj.AddMember("outChannels", info->outChannels, json.GetAllocator());

      auto name = (info == infoIn ? "inInfo" : "outInfo");
      rapidjson::Value key(name, json.GetAllocator());
      json.AddMember(key, obj.Move(), json.GetAllocator());
    }
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);

    std::ofstream amJsonFile(
        GetFullPath(
            FLAGS_acoustic_module_definition_file,
            FLAGS_output_files_base_path),
        std::ios::out);
    std::string str(buffer.GetString(), buffer.GetSize());
    amJsonFile << str;
    amJsonFile.close();

    sequential->reset_buffers();
    torch::save(
        sequential,
        GetFullPath(
            FLAGS_acoustic_module_parameter_file,
            FLAGS_output_files_base_path));
  }
}