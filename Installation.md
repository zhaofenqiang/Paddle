# Build PaddleOnACL's CAPI library  
For now, it has been built successfully on these two platforms
-  [Build for Raspberry Pi]()
-  [Build for android]()

## Build for Raspberry Pi
Following this [guide](https://community.arm.com/graphics/b/blog/posts/cartoonifying-images-on-raspberry-pi-with-the-compute-library). A [Ubuntu Mate](https://ubuntu-mate.org/download/) OS is recommended. And [Etcher](https://etcher.io/) is recommended to burn the OS into SD cards.

#### 1. Cross compile [Arm ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) for Raspberry Pi



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
Copy the `build` folder in `ComputeLibrary` to Raspberry Pi, and run 
```
LD_LIBRARY_PATH=. ./neon_convolution
```
