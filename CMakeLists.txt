cmake_minimum_required(VERSION 3.10)

project(light)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -ffinite-math-only -fno-math-errno -fno-trapping-math -fno-signaling-nans -fno-rounding-math")

include_directories(Boost)

add_executable(light light.cpp)
target_link_libraries(light Eigen3::Eigen Boost::program_options OpenMP::OpenMP_CXX)
