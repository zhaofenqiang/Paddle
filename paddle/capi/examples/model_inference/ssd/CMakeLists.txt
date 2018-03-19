project(ssd CXX C)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
find_package(OpenCV)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

aux_source_directory(. SRC_LIST)
include_directories("/home/zfq/Paddle/build/CAPI/include" ${CMAKE_CURRENT_SOURCE_DIR})
LINK_DIRECTORIES("/home/zfq/Paddle/build/CAPI/lib")  
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} libpaddle_capi_shared.so)
if(OPENCV_FOUND)
	message("opencv founded.")
    target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES})
endif()
