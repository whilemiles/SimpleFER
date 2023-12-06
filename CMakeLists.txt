cmake_minimum_required(VERSION 3.0.0)
project(SimpleFER VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:\\Program Files\\opencv\\build\\x64\\mingw\\lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})
# link_directories("C:\\Program Files\\opencv\\build\\x64\\vc16\\lib")
# other settings
add_executable(SimpleFER src/SimpleFER.cpp)

target_link_libraries(SimpleFER PRIVATE ${OpenCV_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
