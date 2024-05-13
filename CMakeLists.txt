cmake_minimum_required(VERSION 3.0.0)
project(SimpleFER VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

set(RESTBED_INCLUDE_DIR /home/miles/Projects/restbed/distribution/include)
set(RESTBED_LIBRARY_DIR /home/miles/Projects/restbed/distribution/library)
set(PYTHON_LIBRARY /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so)
set(PYTHON_INCLUDE_DIR /home/miles/.conda/envs/FER/include/python3.8)

# Avoid libtorch warning: Failed to compute shorthash for libnvrtc.so
include_directories(${RESTBED_INCLUDE_DIR})
link_directories(${RESTBED_LIBRARY_DIR})

include_directories(${PYTHON_INCLUDE_DIR})
# link_directories(${PYTHON_LIBRARY})

find_package(PythonInterp REQUIRED) 

# Cuda
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc) 
set(CUDACXX /usr/local/cuda-12.1/bin/nvcc)

# OPENCV
find_package(OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIBRARIES})

# Torch
find_package(Torch REQUIRED PATHS "~/libtorch")

find_package(nlohmann_json 3.6.0 REQUIRED)

# Other settings
add_executable(SimpleFER src/SimpleFER/main.cpp src/SimpleFER/FERPipeline.cpp src/SimpleFER/Face.cpp src/SimpleFER/functions.cpp src/SimpleFER/User.cpp)
# add_executable(SimpleFER src/SimpleFER/SimpleFER.cpp src/SimpleFER/FERPipeline.cpp src/SimpleFER/Face.cpp src/SimpleFER/functions.cpp)

target_link_libraries(SimpleFER PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} pthread)
target_link_libraries(SimpleFER PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(SimpleFER PRIVATE restbed)
target_link_libraries(SimpleFER PRIVATE ${PYTHON_LIBRARY})

target_include_directories(SimpleFER PRIVATE ${PYTHON_INCLUDE_DIR})
target_include_directories(SimpleFER PRIVATE ${RESTBED_LIBRARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
