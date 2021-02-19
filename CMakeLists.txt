cmake_minimum_required(VERSION 3.10)

project(light)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -ffinite-math-only -fno-math-errno -fno-trapping-math -fno-signaling-nans -fno-rounding-math")

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#add_definitions(-DUSE_EIGEN)

file(GLOB EXR_SRC ${PROJECT_SOURCE_DIR}/src/exr/*.hpp ${PROJECT_SOURCE_DIR}/src/exr/*.cpp)

add_library(exr ${EXR_SRC})
link_directories(/usr/local/lib)
target_link_libraries(exr OpenMP::OpenMP_CXX IlmImf)

file(GLOB TRACE_SRC ${PROJECT_SOURCE_DIR}/src/*.hpp ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(trace ${TRACE_SRC})
target_link_libraries(trace Eigen3::Eigen Boost::program_options OpenMP::OpenMP_CXX ${OpenCV_LIBS} exr)