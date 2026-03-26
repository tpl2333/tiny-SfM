#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace py = pybind11;

// 证明 Eigen 能用：计算 3x3 矩阵行列式
double det3x3(double m00, double m01, double m02,
              double m10, double m11, double m12,
              double m20, double m21, double m22) {
    Eigen::Matrix3d m;
    m << m00, m01, m02,
         m10, m11, m12,
         m20, m21, m22;
    return m.determinant();
}

PYBIND11_MODULE(ba_core, m) {
    m.def("det3x3", &det3x3, "A test function for Eigen");
}