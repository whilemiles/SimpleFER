cmake_minimum_required(VERSION 3.0.0)
project(SimpleFER VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

find_package(OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})

find_package(Torch REQUIRED)

# other settings
add_executable(SimpleFER src/SimpleFER/SimpleFER.cpp src/SimpleFER/FERPipeline.cpp src/SimpleFER/Face.cpp src/SimpleFER/functions.cpp)

target_link_libraries(SimpleFER PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
