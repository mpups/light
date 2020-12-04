cmake_minimum_required(VERSION 3.10)

project(light)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -ffinite-math-only -fno-math-errno -fno-trapping-math -fno-signaling-nans -fno-rounding-math")

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#add_definitions(-DUSE_EIGEN)
add_executable(trace trace.cpp)
target_link_libraries(trace Eigen3::Eigen Boost::program_options OpenMP::OpenMP_CXX ${OpenCV_LIBS})
