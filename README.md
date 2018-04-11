# PaddleOnACL

### This project is still under development. Sorry for the ugly code and plenty of bugs. Welcome to contribute! 

PaddleOnACL is a work during my arm intern. It aims at porting [paddle](https://github.com/PaddlePaddle/Paddle)'s [CAPI](http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/capi/workflow_of_capi_cn.html) onto [ArmComputeLibrary](https://github.com/ARM-software/ComputeLibrary) instead of MKL or OpenBlas library, seeking for performance gain of deep learning applicaiton at mobile and embedded devices.

For now(2018.04.11), it is based on paddle's develop branch at this [commit](https://github.com/zhaofenqiang/PaddleOnACL/commit/128adf53cb4517f2a4f123044c1ffffd6a3fa74d) and arm ComputeLibrary [v18.03](https://github.com/ARM-software/ComputeLibrary/tree/v18.03).

### Konwn issues:
- Crash when running conv13 of MobileNet, #
- Crash when running conv2_2 of vgg_ssd_net, #
- With only activation and softmax layer enabled, crashed at 2nd iteration


### Pending work:
- [ ] Merge from latest paddle develop branch 
- [ ] Port BatchNormlization layer  
- [ ] Port Sigmoid Layer
- [ ] Port TanH Layer
- [ ] Define Bypass variable to enable and disable ACL layer
- [ ] Add standard and optimized log info
- [ ] Add macro definition to switch between neon/opencl and gemmConv/directConv

### Tutorial
[Installation instructions]()  
[Inference benchmark demo]()  
[Mobile AI Camera App Demo](https://github.com/zhaofenqiang/Mobile/tree/develop/Demo/Android/AICamera)


Benchmark
===============
### Note:
- All the data is tested under Debug mode, it should be smaller when build with Release/MinSizeRel.   
- The unit is millisecond.
- Blank is TODO
- The data was collected at normal inference before crash mentioned [above](https://github.com/zhaofenqiang/PaddleOnACL#konwn-issues)

### Paddle/PaddleOnACL on Raspberry Pi 3 

|   |init	paddle|creat model |1st run 	|2nd~10th avg | Conv |BN |Activation |FC|
| - | :-: | :-: | :-: | :-: |  :-: | :-: | :-: | :-: |
| MobileNet | 3.0/3.0 | 153/163  |   2769/   |  2596/
|SSD      | 3.8/3.0    | 5403/5380 |
| VGG16  |
|AlexNet |
   
(Just found AlexNet example [here](https://github.com/jczaja/test-paddle-c-api))

### Paddle/PaddleOnACL on HUAWEI Mate10 Pro

|   | init	paddle|creat model |1st run | 2nd~10th avg | Conv |  BN | Activation | FC |
| - | :-: | :-: | :-: | :-: |  :-: | :-: | :-: | :-: |
| MobileNet| 0.9/0.9 | 68/ |  | 218/  |   | 
|SSD | 0.8/0.9 | 390/ |  | 6449/
| VGG16|  

