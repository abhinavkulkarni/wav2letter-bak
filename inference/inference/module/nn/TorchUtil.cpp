//
// Created by abhinav on 10/3/20.
//

#include "inference/module/nn/TorchUtil.h"

#include <cereal/external/rapidjson/istreamwrapper.h>
#include <cereal/external/rapidjson/stringbuffer.h>
#include <torch/csrc/api/include/torch/all.h>
#include <utility>

namespace F = torch::nn::functional;

namespace w2l::streaming {

torch::Tensor StackSequentialImpl::forward(torch::Tensor x) {
  return torch::nn::SequentialImpl::forward(x);
}

template <typename... Modules>
StackSequentialImpl::StackSequentialImpl(Modules&&... modules)
    : torch::nn::SequentialImpl(modules...) {}

StackSequentialImpl::StackSequentialImpl() : torch::nn::SequentialImpl() {}

void StackSequentialImpl::pretty_print(std::ostream& stream) const {
  SequentialImpl::pretty_print(stream);
}

void StackSequentialImpl::start() {
  for (auto& module : modules(false))
    if (auto ptr = module->as<Conv1dUnequalPadding>())
      ptr->start();
}

void StackSequentialImpl::finish() {
  for (auto& module : modules(false))
    if (auto ptr = module->as<Conv1dUnequalPadding>())
      ptr->finish();
}

void StackSequentialImpl::reset_buffers() {
  for (auto& module : modules(false))
    if (auto ptr = module->as<Conv1dUnequalPadding>())
      ptr->reset_buffers();
    else if (auto ptr = module->as<ResidualTorch>())
      ptr->reset_buffers();
}

PermuteImpl::PermuteImpl(std::vector<long>&& vec) {
  this->vec = std::move(vec);
  permutation = c10::IntArrayRef(this->vec);
}

torch::Tensor PermuteImpl::forward(torch::Tensor x) {
  x = x.permute(permutation);
  return x;
}

void PermuteImpl::pretty_print(std::ostream& stream) const {
  stream << "Permute(" << permutation << ")";
}

ReshapeImpl::ReshapeImpl(std::vector<long>&& vec) {
  this->vec = std::move(vec);
  sizes = c10::IntArrayRef(this->vec);
}

torch::Tensor ReshapeImpl::forward(torch::Tensor x) {
  x = x.reshape(sizes);
  return x;
}

void ReshapeImpl::pretty_print(std::ostream& stream) const {
  stream << "Reshape(" << sizes << ")";
}

W2LGroupNormImpl::W2LGroupNormImpl(float alpha, float beta)
    : alpha(register_parameter("alpha", torch::tensor(alpha))),
      beta(register_parameter("beta", torch::tensor(beta))) {}

void W2LGroupNormImpl::pretty_print(std::ostream& stream) const {
  stream << "W2LGroupNorm2D("
         << "alpha=" << alpha.item().toFloat()
         << ", beta=" << beta.item().toFloat() << ")";
}

torch::Tensor W2LGroupNormImpl::forward(torch::Tensor x) {
  auto mean = x.mean(1, true);
  auto std = x.std(1, false, true);
  x = (x - mean) / std * alpha + beta;
  return x;
}

ResidualTorchImpl::ResidualTorchImpl(
    std::string name,
    torch::nn::AnyModule anyModule)
    : padding(register_buffer("padding", torch::empty(0))) {
  if (name == "ReLU")
    anyModule = register_module(name, anyModule.get<torch::nn::ReLU>());
  else if (name == "Sequential")
    anyModule = register_module(name, anyModule.get<StackSequential>());
  else if (name == "Identity")
    anyModule = register_module(name, anyModule.get<torch::nn::Identity>());
  else if (name == "Linear")
    anyModule = register_module(name, anyModule.get<torch::nn::Linear>());
  else if (name == "Conv1d")
    anyModule = register_module(name, anyModule.get<Conv1dUnequalPadding>());
  else if (name == "Residual")
    anyModule = register_module(name, anyModule.get<ResidualTorch>());
  else if (name == "GroupNorm")
    anyModule = register_module(name, anyModule.get<W2LGroupNorm>());
  this->name = std::move(name);
  this->anyModule = std::move(anyModule);
}

void ResidualTorchImpl::pretty_print(std::ostream& stream) const {
  stream << "Residual";
}

torch::Tensor ResidualTorchImpl::forward(torch::Tensor x) {
  auto y = anyModule.forward(x);

  int dim = x.sizes().size() == 3 ? -1 : 0;
  x = torch::cat({padding, x}, dim);
  auto size = std::min(x.size(dim), y.size(dim));
  auto z = x.slice(dim, 0, size) + y.slice(dim, 0, size);
  padding = x.slice(dim, size);
  return z;
}

void ResidualTorchImpl::reset_buffers() {
  padding = torch::empty(
      0,
      torch::TensorOptions().device(padding.device()).dtype(padding.dtype()));
}

Conv1dUnequalPaddingImpl::Conv1dUnequalPaddingImpl(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    int leftPadding,
    int rightPadding,
    int groups)
    : torch::nn::Conv1dImpl(torch::nn::Conv1dOptions(
                                inChannels / groups,
                                outChannels / groups,
                                kernelSize)
                                .stride(stride)
                                .groups(1)),
      leftPadding(leftPadding),
      rightPadding(rightPadding),
      groups(groups),
      leftPaddingTensor(register_buffer("leftPaddingTensor", torch::empty(0))),
      rightPaddingTensor(
          register_buffer("rightPaddingTensor", torch::empty(0))) {}

torch::Tensor Conv1dUnequalPaddingImpl::forward(torch::Tensor x) {
  x = torch::cat({leftPaddingTensor, x, rightPaddingTensor}, -1);
  int kernelSize = options.kernel_size()->at(0);
  int stride = options.stride()->at(0);
  int nOutFrames = (x.size(-1) - kernelSize) / stride + 1;
  int consumedFrames = nOutFrames * stride;
  int inChannels = options.in_channels();
  leftPaddingTensor = x.slice(-1, consumedFrames);

  auto y = torch::empty(
      0,
      torch::TensorOptions()
          .device(leftPaddingTensor.device())
          .dtype(leftPaddingTensor.dtype()));
  for (int i = 0; i < groups; i++) {
    auto x_ = torch::nn::Conv1dImpl::forward(
        x.slice(1, i * inChannels, (i + 1) * inChannels));
    y = torch::cat({y, x_}, 1);
  }

  return y;
}

void Conv1dUnequalPaddingImpl::pretty_print(std::ostream& stream) const {
  torch::nn::Conv1dImpl::pretty_print(stream);
  if (leftPadding and rightPadding)
    stream << "\b, padding=(" << leftPadding << ", " << rightPadding << "))";
}

void Conv1dUnequalPaddingImpl::start() {
  leftPaddingTensor = torch::zeros(
      {1, options.in_channels() * groups, leftPadding},
      torch::TensorOptions()
          .device(leftPaddingTensor.device())
          .dtype(leftPaddingTensor.dtype()));
}

void Conv1dUnequalPaddingImpl::finish() {
  rightPaddingTensor = torch::zeros(
      {1, options.in_channels() * groups, rightPadding},
      torch::TensorOptions()
          .device(rightPaddingTensor.device())
          .dtype(rightPaddingTensor.dtype()));
}

void Conv1dUnequalPaddingImpl::reset_buffers() {
  leftPaddingTensor = torch::empty(
      0,
      torch::TensorOptions()
          .device(leftPaddingTensor.device())
          .dtype(leftPaddingTensor.dtype()));
  rightPaddingTensor = torch::empty(
      0,
      torch::TensorOptions()
          .device(rightPaddingTensor.device())
          .dtype(rightPaddingTensor.dtype()));
}

std::tuple<
    std::shared_ptr<InferenceModuleInfo>,
    std::shared_ptr<InferenceModuleInfo>,
    StackSequential>
getTorchModule(const std::shared_ptr<Sequential>& module) {
  torch::NoGradGuard no_grad;
  auto tuple = module->getTorchModule();
  const auto& [type, infoIn, infoOut, anyModule] = tuple;
  std::map<std::string, int> counts;

  auto getName = [](const std::string& name,
                    std::map<std::string, int>& counts) {
    if (counts.count(name) == 0)
      counts[name] = 0;
    return name + "-" + std::to_string(counts[name]++);
  };

  auto sequential = StackSequential();

  if (infoIn->inShape == InferenceModuleInfo::shape::SHAPE_2D) {
    std::vector<long> shape = {-1, infoOut->inChannels};
    sequential->push_back(
        getName("Reshape", counts), Reshape(std::move(shape)));
  } else if (infoIn->inShape == InferenceModuleInfo::shape::SHAPE_3D) {
    std::vector<long> shape = {1, -1, infoIn->inChannels};
    sequential->push_back(
        getName("Reshape", counts), Reshape(std::move(shape)));
    std::vector<long> permutation = {0, 2, 1};
    sequential->push_back(
        getName("Permute", counts), Permute(std::move(permutation)));
  }

  if (anyModule.ptr()->as<StackSequential>()) {
    auto seqModule = anyModule.get<StackSequential>();
    std::vector<std::string> names;
    for (auto&& item : seqModule->named_children()) {
      auto name = item.key();
      name = name.substr(0, name.find('-'));
      names.push_back(name);
    }
    int i = 0;
    for (auto itr = seqModule->begin(); itr != seqModule->end(); itr++)
      sequential->push_back(getName(names[i++], counts), *itr);
  } else
    sequential->push_back(getName(type, counts), anyModule);

  if (infoOut->outShape == InferenceModuleInfo::shape::SHAPE_2D) {
  } else if (infoOut->outShape == InferenceModuleInfo::shape::SHAPE_3D) {
    std::vector<long> permutation = {0, 2, 1};
    sequential->push_back(
        getName("Permute", counts), Permute(std::move(permutation)));
  }

  return {infoIn, infoOut, sequential};
}

torch::nn::AnyModule getTorchModule_(rapidjson::Document& obj) {
  auto name = std::string(obj["name"].GetString());
  if (name == "ReLU")
    return torch::nn::AnyModule(torch::nn::ReLU());
  else if (name == "Identity")
    return torch::nn::AnyModule(torch::nn::Identity());
  else if (name == "Linear") {
    auto inFeatures = obj["inFeatures"].GetInt(),
         outFeatures = obj["outFeatures"].GetInt();
    auto linear = torch::nn::Linear(inFeatures, outFeatures);
    return torch::nn::AnyModule(linear);
  } else if (name == "Conv1d") {
    auto inChannels = obj["inChannels"].GetInt(),
         outChannels = obj["outChannels"].GetInt(),
         kernelSize = obj["kernelSize"].GetInt(),
         groups = obj["groups"].GetInt(), stride = obj["stride"].GetInt(),
         leftPadding = obj["leftPadding"].GetInt(),
         rightPadding = obj["rightPadding"].GetInt();
    auto conv1d = Conv1dUnequalPadding(
        inChannels,
        outChannels,
        kernelSize,
        stride,
        leftPadding,
        rightPadding,
        groups);
    return torch::nn::AnyModule(conv1d);
  } else if (name == "GroupNorm") {
    auto alpha = obj["alpha"].GetFloat(), beta = obj["beta"].GetFloat();
    return torch::nn::AnyModule(W2LGroupNorm(alpha, beta));
  } else if (name == "Permute") {
    std::vector<long> permutation;
    for (auto&& item : obj["permutation"].GetArray())
      permutation.push_back(item.GetInt());
    auto permute = Permute(std::move(permutation));
    return torch::nn::AnyModule(permute);
  } else if (name == "Reshape") {
    std::vector<long> shape;
    for (auto&& item : obj["shape"].GetArray())
      shape.push_back(item.GetInt());
    auto reshape = Reshape(std::move(shape));
    return torch::nn::AnyModule(reshape);
  } else if (name == "Residual") {
    rapidjson::Document moduleObj;
    moduleObj.CopyFrom(obj["module"], obj.GetAllocator());
    name = obj["module"].GetObject()["name"].GetString();
    auto residual = ResidualTorch(name, getTorchModule_(moduleObj));
    return torch::nn::AnyModule(residual);
  } else {
    auto sequential = getTorchModule(obj);
    return torch::nn::AnyModule(sequential);
  }
}

StackSequential getTorchModule(const rapidjson::Document& json) {
  std::map<std::string, int> counts;

  auto getName = [](const std::string& name,
                    std::map<std::string, int>& counts) {
    if (counts.count(name) == 0)
      counts[name] = 0;
    return name + "-" + std::to_string(counts[name]++);
  };

  StackSequential sequential;

  for (auto& child : json["children"].GetArray()) {
    rapidjson::Document obj;
    obj.CopyFrom(child, obj.GetAllocator());
    auto name = std::string(obj["name"].GetString());
    name = getName(name, counts);
    auto anyModule = getTorchModule_(obj);
    sequential->push_back(name, anyModule);
  }

  return sequential;
}

rapidjson::Document getJSON(const std::shared_ptr<InferenceModule>& dnnModule) {
  rapidjson::Document d(rapidjson::kObjectType);
  auto& allocator = d.GetAllocator();

  for (auto& m : dnnModule->getJSON(allocator).GetObject())
    d.AddMember(m.name, m.value.Move(), allocator);

  return d;
}

rapidjson::Document getJSON_(
    const std::string& name,
    const torch::nn::AnyModule& anyModule,
    rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Document d(rapidjson::kObjectType);

  d.AddMember(
      "name",
      rapidjson::Value(name.c_str(), name.size(), allocator),
      allocator);

  if (name == "Linear") {
    auto& options = anyModule.ptr()->as<torch::nn::Linear>()->options;
    d.AddMember("inFeatures", options.in_features(), allocator);
    d.AddMember("outFeatures", options.out_features(), allocator);
  } else if (name == "Conv1d") {
    auto* ptr = anyModule.ptr()->as<Conv1dUnequalPadding>();
    auto& options = ptr->options;
    int groups = ptr->groups;
    d.AddMember("inChannels", options.in_channels() * groups, allocator);
    d.AddMember("outChannels", options.out_channels() * groups, allocator);
    d.AddMember("kernelSize", options.kernel_size()->at(0), allocator);
    d.AddMember("groups", groups, allocator);
    d.AddMember("stride", options.stride()->at(0), allocator);
    d.AddMember("leftPadding", ptr->leftPadding, allocator);
    d.AddMember("rightPadding", ptr->rightPadding, allocator);
  } else if (name == "GroupNorm") {
    auto* ptr = anyModule.ptr()->as<W2LGroupNorm>();
    d.AddMember("alpha", ptr->alpha.item().toFloat(), allocator);
    d.AddMember("beta", ptr->beta.item().toFloat(), allocator);
  } else if (name == "Residual") {
    auto* ptr = anyModule.ptr()->as<ResidualTorch>();
    d.AddMember(
        "module",
        getJSON_(ptr->name, ptr->anyModule, allocator).Move(),
        allocator);
  } else if (name == "Permute") {
    auto* ptr = anyModule.ptr()->as<Permute>();
    rapidjson::Document permutation(rapidjson::kArrayType);
    for (auto&& i : ptr->permutation)
      permutation.PushBack(i, allocator);
    d.AddMember("permutation", permutation, allocator);
  } else if (name == "Reshape") {
    auto* ptr = anyModule.ptr()->as<Reshape>();
    rapidjson::Document shape(rapidjson::kArrayType);
    for (auto&& i : ptr->sizes)
      shape.PushBack(i, allocator);
    d.AddMember("shape", shape, allocator);
  } else if (name == "Sequential") {
    auto* ptr = anyModule.ptr()->as<StackSequential>();
    rapidjson::Document children(rapidjson::kArrayType);
    std::vector<std::string> names;
    for (auto&& item : ptr->named_children()) {
      auto name = item.key();
      name = name.substr(0, name.find('-'));
      names.push_back(name);
    }
    int i = 0;
    for (auto& itr : *ptr)
      children.PushBack(getJSON_(names[i++], itr, allocator).Move(), allocator);
    d.AddMember("children", children, allocator);
  }

  return d;
}

rapidjson::Document getJSON(const StackSequential& seqModule) {
  rapidjson::Document d(rapidjson::kObjectType);
  auto& allocator = d.GetAllocator();

  for (auto& m :
       getJSON_("Sequential", torch::nn::AnyModule(seqModule), allocator)
           .GetObject())
    d.AddMember(m.name, m.value.Move(), allocator);

  return d;
}

std::tuple<
    std::shared_ptr<InferenceModuleInfo>,
    std::shared_ptr<InferenceModuleInfo>,
    StackSequential>
loadTorchModule(
    const std::string& acoustic_module_definition_file,
    const std::string& acoustic_module_parameter_file,
    const std::string& acoustic_module_precision) {
  rapidjson::Document json;
  std::ifstream amDefinitionFile(acoustic_module_definition_file, std::ios::in);
  rapidjson::IStreamWrapper isw(amDefinitionFile);
  json.ParseStream(isw);

  auto sequential = getTorchModule(json);
  std::shared_ptr<InferenceModuleInfo> infoIn, infoOut;

  auto dtype =
      (acoustic_module_precision == "fp16") ? torch::kFloat16 : torch::kFloat;
  sequential->to(dtype);
  torch::load(sequential, acoustic_module_parameter_file);
  if (not torch::cuda::is_available())
    sequential->to(torch::kFloat);

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

  return {infoIn, infoOut, sequential};
}

} // namespace w2l::streaming
