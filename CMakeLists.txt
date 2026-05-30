cmake_minimum_required(VERSION 3.10)
project(BA_Backend)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(ba_core src/ba_core.cpp)

target_link_libraries(ba_core PRIVATE Ceres::ceres Eigen3::Eigen)
