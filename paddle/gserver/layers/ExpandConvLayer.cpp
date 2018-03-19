/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "ExpandConvLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

DEFINE_bool(use_nnpack,
            false,
            "Whether to use nnpack for convolution calculation.");

namespace paddle {

/*
 * The calculation of the exconvt(convolution transpose (deconv) operation)
 * is a swap of forward and backward of the calculation of exconv.
 * */
REGISTER_LAYER(exconv, ExpandConvLayer);
REGISTER_LAYER(exconvt, ExpandConvLayer);

inline bool isDepthwiseConv(int channels, int groups) {
  return channels == groups;
}

bool ExpandConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  int index = 0;
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();

    // create a new weight
    size_t height, width;
    height = filterPixels_[index] * filterChannels_[index];
    width = (!isDeconv_) ? numFilters_ : channels_[index];
    CHECK_EQ(parameters_[index]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[index]);
    weights_.emplace_back(w);
    index++;
  }

  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ = std::unique_ptr<Weight>(
          new Weight(1, numFilters_, biasParameter_, 0));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_, 0));
    }
  }

  getOutputSize();

  size_t numInputs = config_.inputs_size();
  inputShape_.resize(numInputs);
  filterShape_.resize(numInputs);
  outputShape_.resize(numInputs);

//  std::string convType;
//  std::string convGradInputType;
//  std::string convGradFilterType;

 // zfq:Deprecated: Convolution Layer uses the ACLConv function by default in the following forward()

//  for (int i = 0; i < config_.inputs_size(); i++) {
//    std::vector<size_t> paddings = {(size_t)paddingY_[i], (size_t)padding_[i]};
//    std::vector<size_t> strides = {(size_t)strideY_[i], (size_t)stride_[i]};
//    std::vector<size_t> dilations = {(size_t)dilationY_[i],
//                                     (size_t)dilation_[i]};
//
//    bool useDilation = ((size_t)dilationY_[i] > 1 || (size_t)dilation_[i] > 1);
//
//    convType = "ACLConv";
//    convGradInputType = "GemmConvGradInput";
//    convGradFilterType = "GemmConvGradFilter";

    //zfq: use NEON for  conv
//    // If depth wise convolution and useGpu == true
//    if (useGpu_ && isDepthwiseConv(channels_[i], groups_[i]) && !isDeconv_) {
//      convType = "DepthwiseConv";
//      convGradInputType = "DepthwiseConvGradInput";
//      convGradFilterType = "DepthwiseConvGradFilter";
//    }
//
//    // If depth wise convolution and useGpu == false and ARM-NEON
//    if (!useGpu_ && isDepthwiseConv(channels_[i], groups_[i]) && !isDeconv_) {
//#if defined(__ARM_NEON__) || defined(__ARM_NEON)
//      if ((filterSize_[i] == filterSizeY_[i]) &&
//          (filterSize_[i] == 3 || filterSize_[i] == 4) &&
//          (stride_[i] == strideY_[i]) && (stride_[i] == 1 || stride_[i] == 2) &&
//          !useDilation) {
//        convType = "NeonDepthwiseConv";
//      }
//#endif
//    }

//    if (FLAGS_use_nnpack && !isDeconv_ && !useDilation) {
//      createFunction(forward_,
//                     "NNPACKConv",
//                     FuncConfig()
//                         .set("paddings", paddings)
//                         .set("strides", strides)
//                         .set("groups", (size_t)groups_[i])
//                         .set("algo", std::string("auto")));
//    } else {
//      createFunction(forward_,
//                     !isDeconv_ ? convType : convGradInputType,
//                     FuncConfig()
//                         .set("paddings", paddings)
//                         .set("strides", strides)
//                         .set("dilations", dilations)
//                         .set("groups", (size_t)groups_[i]));

      //Only do inference.
//      createFunction(backward_,
//                     !isDeconv_ ? convGradInputType : convType,
//                     FuncConfig()
//                         .set("paddings", paddings)
//                         .set("strides", strides)
//                         .set("dilations", dilations)
//                         .set("groups", (size_t)groups_[i]));
//
//      createFunction(backward_,
//                     convGradFilterType,
//                     FuncConfig()
//                         .set("paddings", paddings)
//                         .set("strides", strides)
//                         .set("dilations", dilations)
//                         .set("groups", (size_t)groups_[i]));
//    }
//  }
  return true;
}

size_t ExpandConvLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  return layerSize;
}

// i is the index of input layers
#define BACKWARD_INPUT(i, inputs, outputs) \
  backward_[2 * i]->calc(inputs, outputs)
#define BACKWARD_FILTER(i, inputs, outputs) \
  backward_[2 * i + 1]->calc(inputs, outputs)

void ExpandConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getOutputSize());

  // Calculate the shape of the input, output, and filter.
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
	arm_compute::TensorShape input_shape((unsigned int)imgSizeW_[i], (unsigned int)imgSizeH_[i], (unsigned int)channels_[i], (unsigned int)batchSize);
	arm_compute::PadStrideInfo conv_info(this->stride_[i], this->strideY_[i], this->padding_[i], this->paddingY_[i]);
	arm_compute::TensorShape weights_shape((unsigned int)filterSize_[i], (unsigned int)filterSizeY_[i], (unsigned int)channels_[i], (unsigned int)numFilters_);
	arm_compute::TensorShape biases_shape ((unsigned int)numFilters_);
	arm_compute::TensorShape output_shape((unsigned int)outputW_[i], (unsigned int)outputH_[i],(unsigned int)numFilters_,(unsigned int)batchSize);

    real* inputData = getInputValue(i)->getData();
    real* filterData = weights_[i]->getW()->getData();
    real* outputData = getOutputValue()->getData();

    //[kernel_x, kernel_y, IFM, OFM]
    new_tensor(weights(), weights_shape, filterData);
    if (biases_.get()) {
    	real* biasData = (*biases_->getW()).getData();
        new_tensor(biases(), biases_shape, biasData);
    }
 //[width, height, IFM]
    new_tensor(input(), input_shape, inputData);
 //[width, height, OFM]
    new_tensor(output(),output_shape,outputData);
    acl_configure(conv, this, conv_info);

    this->acl_run(inputData, outputData);

  }
  /* activation */
  forwardActivation();
}

void ExpandConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr outGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1, sharedBiases_);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  // Calculate the input grad and filter grad.
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    if (getInputGrad(i)) {
      BufferArgs inputs;
      BufferArgs outputs;
      inputs.addArg(*getOutputGrad(), outputShape_[i]);
      inputs.addArg(*weights_[i]->getW(), filterShape_[i]);
      outputs.addArg(*getInputGrad(i), inputShape_[i], ADD_TO);
      BACKWARD_INPUT(i, inputs, outputs);
    }

    if (weights_[i]->getWGrad()) {
      BufferArgs inputs;
      BufferArgs outputs;
      if (!isDeconv_) {
        inputs.addArg(*getOutputGrad(), outputShape_[i]);
        inputs.addArg(*getInputValue(i), inputShape_[i]);
      } else {
        inputs.addArg(*getInputValue(i), inputShape_[i]);
        inputs.addArg(*getOutputGrad(), outputShape_[i]);
      }
      outputs.addArg(*weights_[i]->getWGrad(), filterShape_[i], ADD_TO);
      BACKWARD_FILTER(i, inputs, outputs);

      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

}  // namespace paddle
