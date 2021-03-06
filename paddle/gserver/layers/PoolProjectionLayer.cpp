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

#include "PoolProjectionLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

size_t PoolProjectionLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = imgSizeY_;
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = imgSize_;
  }

  outputH_ = outputSize(imgSizeH_,
                        sizeY_,
                        confPaddingY_,
                        strideY_,
                        /* caffeMode */ false);
  outputW_ = outputSize(imgSizeW_,
                        sizeX_,
                        confPadding_,
                        stride_,
                        /* caffeMode */ false);

  layerSize = outputH_ * outputW_ * channels_;

  return layerSize;
}

void PoolProjectionLayer::forward(PassType passType) {
  Layer::forward(passType);
  const Argument& in = getInput(0);
  int batchSize = in.value->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  //call acl function
  arm_compute::TensorShape in_shape ((unsigned int)imgSize_, (unsigned int)imgSizeY_,(unsigned int)channels_);
  arm_compute::TensorShape out_shape((unsigned int)outputX_, (unsigned int)outputY_,(unsigned int)channels_);
  // Initialize ACL.
  arm_compute::PoolingLayerInfo pool_info;
  if(poolType_ == "max-projection")
     pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, stride_, arm_compute::PadStrideInfo(stride_, strideY_, confPadding_, confPaddingY_, arm_compute::DimensionRoundingType::CEIL));
  else
     pool_info=arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, stride_, arm_compute::PadStrideInfo(stride_, strideY_, confPadding_, confPaddingY_, arm_compute::DimensionRoundingType::CEIL));

  real* inputData = in.value->getData();
  real* outputData =  output_.value->getData();

  new_tensor(input(),in_shape,  inputData);
  new_tensor(output(),out_shape, outputData);
  acl_configure(pooling, this, pool_info);

  acl_run(inputData, outputData);

}

void PoolProjectionLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  if (NULL == getInputGrad(0)) {
    return;
  }
  poolProjection_->backward(callback);
}
}  // namespace paddle
