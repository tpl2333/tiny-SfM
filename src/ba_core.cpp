#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <iostream>

namespace py = pybind11;

// SIMPLE_PINHOLE
struct SnavelyReprojectionError {

    // 构造时传入观测到的像素坐标 [u, v]
    SnavelyReprojectionError(double obsx, double obsy, double cx, double cy)
        : obs_x(obsx), obs_y(obsy), c_x(cx), c_y(cy) {}

    template <typename T>
    bool operator()(const T* const camera, // SIMPLE_PINHOLE [r1,r2,r3,t1,t2,t3,f]
                    const T* const point,  // mapppoint coordination [X, Y, Z]
                    T* residuals) const {

        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        const T& f = camera[6];
        T pred_u = f * xp + T(c_x);
        T pred_v = f * yp + T(c_y);

        residuals[0] = pred_u - T(obs_x);
        residuals[1] = pred_v - T(obs_y);
        return true;
    }

    double obs_x, obs_y, c_x, c_y;
};

void solve_ba(
    py::array_t<double> cameras,          // (N, 7) - [r1,r2,r3, t1,t2,t3, f]
    py::array_t<double> points,         // (M, 3) - [X,Y,Z]
    py::array_t<double> observations,    // (K, 2) - [u, v]
    py::array_t<int> camera_indices,      // (K,)   - 每条观测对应的相机 ID
    py::array_t<int> point_indices,     // (K,)   - 每条观测对应的点 ID
    double c_x, double c_y,              // 固定主点坐标
    int fixed_camera_idx = 0               
) {
    // 1. 获取原始指针 (直接操作内存，实现零拷贝)
    auto p_cameras = cameras.mutable_data();
    auto p_points = points.mutable_data();
    auto p_obs = observations.data();
    auto p_camera_idx = camera_indices.data();
    auto p_point_idx = point_indices.data();

    // 2. 实例化 Ceres 问题
    ceres::Problem problem;

    // 鲁棒核函数：防止坏匹配（Outliers）拉偏整个优化过程
    // 如果投影误差超过 1 像素，它会降低权重
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    // 3. 核心循环：将所有观测添加为残差块
    for (int i = 0; i < camera_indices.size(); ++i) {
        // 创建代价函数 (自动求导模式)
        // 参数含义: <结构体, 残差维度2, 相机参数维度7, 点参数维度3>
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 7, 3>(
                new SnavelyReprojectionError(p_obs[2*i], p_obs[2*i+1], c_x, c_y));

        // 获取当前观测对应的相机和点在内存里的位置
        double* current_camera = p_cameras + p_camera_idx[i] * 7;
        double* current_point = p_points + p_point_idx[i] * 3;

        problem.AddResidualBlock(cost_function, loss_function, current_camera, current_point);
    }

    if (fixed_camera_idx >= 0 && fixed_camera_idx < cameras.shape(0)) {
        double* fix_cam_ptr = p_cameras + fixed_camera_idx * 7;
        problem.SetParameterBlockConstant(fix_cam_ptr);
        std::cout << "[Ceres] Camera at index " << fixed_camera_idx << " is fixed (Constant)." << std::endl;
    }

    // 4. 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; 
    options.minimizer_progress_to_stdout = true;     
    options.max_num_iterations = 100;                

    // 5. 启动优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

PYBIND11_MODULE(ba_core, m) {
    m.def("solve_ba", &solve_ba, "Solve BA with fixed camera option",
          py::arg("cameras"), py::arg("points"), py::arg("observations"),
          py::arg("camera_indices"), py::arg("point_indices"),
          py::arg("c_x"), py::arg("c_y"), py::arg("fixed_camera_idx") = 0);
}

