# PaddleOnACL

### This project is still under development. Welcome to contribute! And sorry for the ugly code and plenty of bugs.

PaddleOnACL is a work during arm intern. It aims at porting [paddle](https://github.com/PaddlePaddle/Paddle)'s [CAPI](http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/capi/workflow_of_capi_cn.html) onto [ArmComputeLibrary](https://github.com/ARM-software/ComputeLibrary) instead of MKL or OpenBlas library, seeking for performance gain of deep learning applicaiton at mobile or embedded devices.

For now(2018.04.11), it is based on paddle's develop branch at this [commit](https://github.com/zhaofenqiang/PaddleOnACL/commit/128adf53cb4517f2a4f123044c1ffffd6a3fa74d) and arm ComputeLibrary [v18.03](https://github.com/ARM-software/ComputeLibrary/tree/v18.03).

### Konwn issues:
- Crash when running conv13 of MobileNet, #
- Crash when running conv2_2 of vgg_ssd_net, #


### Pending work:
- [ ] Merge from latest paddle develop branch 
- [ ] Port BatchNormlization layer  
- [ ] Port Sigmoid Layer
- [ ] Port TanH Layer
- [ ] Define Bypass variable to enable and disable ACL layer
- [ ] Add standard and optimized log info
- [ ] Add macro definition to switch between neon/opencl and gemmConv/directConv
