cmake_minimum_required(VERSION 3.10)

project(light)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(Boost)

add_executable(light light.cpp)
target_link_libraries(light Eigen3::Eigen Boost::program_options)
