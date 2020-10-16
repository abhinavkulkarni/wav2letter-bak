/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/backend/fbgemm/LinearFbGemm.h"

#include <torch/csrc/api/include/torch/nn/modules/linear.h>
#include <sstream>
#include <stdexcept>

#include "inference/common/IOBuffer.h"

namespace w2l {
namespace streaming {

LinearFbGemm::LinearFbGemm(
    int nInput,
    int nOutput,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias)
    : Linear(nInput, nOutput), bias_(bias) {
  if (!weights || !bias) {
    std::stringstream ss;
    ss << "Invalid arg at LinearFbGemm::LinearFbGemm(nInput=" << nInput
       << " nOutput=" << nOutput
       << " weights=" << (weights ? weights->debugString() : "nullptr")
       << " bias=" << (bias ? bias->debugString() : "nullptr") << ")";
    throw std::invalid_argument(ss.str());
  }

  init(weights);
}

LinearFbGemm::LinearFbGemm() : Linear(0, 0) {}

void LinearFbGemm::init(std::shared_ptr<ModuleParameter> weights) {
  constexpr float alpha = 1.0;
  packedWeights_ = std::make_shared<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::NoTranspose,
      nInput_, // k
      nOutput_, // n
      alpha,
      weights->buffer_.data<float>());
}

std::string LinearFbGemm::debugString() const {
  return debugStringImpl(false);
}

std::string LinearFbGemm::debugStringWithContent() const {
  return debugStringImpl(true);
}

std::string LinearFbGemm::debugStringImpl(bool withContent) const {
  std::stringstream ss;
  ss << "LinearFbGemm:{base=" << Linear::debugString() << " packedWeights_="
     << (packedWeights_
             ? w2l::streaming::debugString(*packedWeights_, withContent)
             : "nullptr")
     << "} bias_=" << (bias_ ? bias_->debugString() : "nullptr") << "}";
  return ss.str();
}

std::shared_ptr<ModuleProcessingState> LinearFbGemm::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  int nFrames = inputBuf->size<float>() / nInput_;
  if (nFrames == 0) {
    return output;
  }
  assert(output->buffers().size() == 1);
  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  const int outSize = nFrames * nOutput_;
  outputBuf->ensure<float>(outSize);
  auto* outPtr = outputBuf->tail<float>();
  for (int i = 0; i < nFrames; ++i) {
    std::copy_n(bias_->buffer_.data<float>(), nOutput_, outPtr + i * nOutput_);
  }

  outputBuf->move<float>(outSize);

  constexpr float beta = 1.0;
  cblas_gemm_compute(
      fbgemm::matrix_op_t::Transpose,
      nFrames,
      inputBuf->data<float>(),
      *packedWeights_,
      beta,
      outPtr);

  inputBuf->consume<float>(nFrames * nInput_);
  return output;
}

std::shared_ptr<InferenceModuleTorchHolder> LinearFbGemm::getTorchModule()
    const {
  at::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kFloat16)); // default dtype to float16
  auto linear = torch::nn::Linear(nInput_, nOutput_);

  auto &weight = linear->weight, &bias = linear->bias;
  // weight = torch::_cast_Half(weight); bias = torch::_cast_Half(bias); // float to float16

  auto fbgemmMat = packedWeights_->pmat();
  for (int i = 0; i < nInput_; i++)
    for (int j = 0; j < nOutput_; j++) {
      auto item = fbgemmMat[packedWeights_->addr(i, j)];
      auto v = fbgemm::cpu_half2float(item);
      at::Half w = v;
      // weight[j][i] = v;
      weight[j][i] = w;
    }

  auto bias_f = bias_->buffer_.data<float>();
  // std::copy_n(bias_->buffer_.data<float>(), nOutput_, bias.data_ptr<float>());
  std::copy_n(bias_f, nOutput_, bias.data_ptr<at::Half>()); // float to float16
  auto holder = std::make_shared<InferenceModuleTorchHolder>(
      "Linear",
      InferenceModuleTorchHolder::shape::SHAPE_2D,
      nInput_,
      InferenceModuleTorchHolder::shape::SHAPE_2D,
      nOutput_,
      torch::nn::AnyModule(linear));
  return holder;
}

std::shared_ptr<Linear> createLinear(
    int nInput,
    int nOutput,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias) {
  return std::make_shared<LinearFbGemm>(nInput, nOutput, weights, bias);
}

} // namespace streaming
} // namespace w2l
