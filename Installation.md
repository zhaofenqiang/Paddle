# Build PaddleOnACL's CAPI library  
For now, it has been built successfully on these two platforms
-  [Build for Raspberry Pi](https://github.com/zhaofenqiang/PaddleOnACL/blob/develop/Installation.md#build-for-raspberry-pi)
-  [Build for android]()

# Build for Raspberry Pi
A [Ubuntu Mate](https://ubuntu-mate.org/download/) OS for Raspberry Pi is recommended. And [Etcher](https://etcher.io/) is recommended to burn the OS into SD cards. 

### 1. Cross compile [Arm ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) for Raspberry Pi 
Following this [guide](https://community.arm.com/graphics/b/blog/posts/cartoonifying-images-on-raspberry-pi-with-the-compute-library). At host machine, type these commands:
```
# Install dependencies (scons, Arm cross-compiler toolchain)
sudo apt-get install git scons gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf 

# Download ACL
git clone https://github.com/ARM-software/ComputeLibrary

# Enter ComputeLibrary folder
cd ComputeLibrary 

# Cross compile the library and the examples
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 os=linux arch=armv7a examples=1
```
After compiling, you can find the dynamic library `libarm_compute.so` `libarm_compute_core.so` in the `build` folder  
Copy the `build` folder to Raspberry Pi, and run at Raspberry Pi: 
```
cd build
LD_LIBRARY_PATH=. ./examples/neon_convolution
```
There should be "Test passed" printed if ACL is built successfully.

### 2. Cross compile PaddleOnACL for Raspberry Pi
```
# Download PaddleOnACL
git clone https://github.com/zhaofenqiang/PaddleOnACL
```
Copy ACL's head files and .so to paddleOnACL, you can use the script at [CaffeHRT](https://github.com/OAID/Caffe-HRT/blob/master/acl_openailab/installation.md#32-build-acl-)
```
cd ComputeLibrary
wget ftp://ftp.openailab.net/tools/script/Computelibrary/Makefile
# replace the install directory with paddleOnACL root path firstly
sudo make install  
```
Make sure that you have the `ComputeLibrary` folder under PaddleOnACL.  
Build PaddleOnACL just following paddle's official [guide](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_raspberry_en.md), my command is:  
```
cd PaddleOnACL
mkdir build
cd build
cmake -DCMAKE_SYSTEM_NAME=RPi -DRPI_ARM_NEON=ON -DCMAKE_INSTALL_PREFIX=CAPI_RPI -DWITH_GPU=OFF -DWITH_C_API=ON  -DWITH_PYTHON=OFF -DWITH_SWIG_PY=OFF -DCMAKE_BUILD_TYPE=Debug ..
sudo make
sudo make install
```
The `libpaddle_capi_shared.so` should be found at `your/path/to/PaddleOnACL/build/CAPI_RPI/lib.

# Build for android
### 1. Cross compile [Arm ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) for android
Following the official [guide](https://arm-software.github.io/ComputeLibrary/v18.03/). At host machine, download the NDK r16b from [here](https://developer.android.com/ndk/downloads/index.html).  
Generate the 32 and/or 64 toolchains by running the following commands:
```
$NDK/build/tools/make_standalone_toolchain.py --arch arm64 --install-dir $MY_TOOLCHAINS/aarch64-linux-android-ndk-r16b --stl gnustl --api 21   
$NDK/build/tools/make_standalone_toolchain.py --arch arm --install-dir $MY_TOOLCHAINS/arm-linux-android-ndk-r16b --stl gnustl --api 21
```
Make sure to add the toolchains to your PATH: 
```
export PATH=$PATH:$MY_TOOLCHAINS/aarch64-linux-android-4.9/bin:$MY_TOOLCHAINS/arm-linux-androideabi-4.9/bin
```

```
# Download ACL
git clone https://github.com/ARM-software/ComputeLibrary

# Enter ComputeLibrary folder
cd ComputeLibrary 

# Cross-compile the library for Android 64bit
CXX=clang++ CC=clang scons Werror=1 -j4 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=android arch=arm64-v8a
```
After compiling, you can find the dynamic library `libarm_compute.so` and static library `libarm_compute-static.a` in the `build` folder  
### 2. Cross compile PaddleOnACL for android
```
# Download PaddleOnACL
git clone https://github.com/zhaofenqiang/PaddleOnACL
```
Copy ACL's head files and .so and .a to paddleOnACL, you can use the script at [CaffeHRT](https://github.com/OAID/Caffe-HRT/blob/master/acl_openailab/installation.md#32-build-acl-)
```
cd ComputeLibrary
wget ftp://ftp.openailab.net/tools/script/Computelibrary/Makefile
# replace the install directory with paddleOnACL root path firstly
sudo make install  
```
Make sure that you have the `ComputeLibrary` folder under PaddleOnACL.  
Build PaddleOnACL just following paddle's official [guide](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_en.md), the [docker](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_en.md#cross-compiling-using-docker) way is highly recommended. My command is:  
```
cd PaddleOnACL
docker build -t paddle:acl-android . -f Dockerfile.android
docker run -it --rm -v $PWD:/paddle -e "ANDROID_ABI=arm64-v8a" -e "ANDROID_API=21" paddle:acl-android
```
The `libpaddle_capi_shared.so` and `libpaddle_capi_whole.a` should be found at `your/path/to/PaddleOnACL/install_android/lib`.


