cmake_minimum_required(VERSION 3.10)
project(BA_Backend)

# 1. 基础设置
set(CMAKE_CXX_STANDARD 14)

# 2. 寻找依赖 (vcpkg 会自动处理路径)
find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# 3. 定义输出模块
# 模块名叫 ba_core，源代码在 src/ba_core.cpp
pybind11_add_module(ba_core src/ba_core.cpp)

# 4. 连接库
# 给 ba_core 装上 Ceres 发动机
target_link_libraries(ba_core PRIVATE Ceres::ceres Eigen3::Eigen)