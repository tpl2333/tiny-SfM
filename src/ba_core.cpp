#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <iostream>

namespace py = pybind11;

// SIMPLE_PINHOLE: shared focal
struct SnavelySharedFocalError {

    // 构造时传入观测到的像素坐标 [u, v]
    SnavelySharedFocalError(double obsx, double obsy, double cx, double cy)
        : obs_x(obsx), obs_y(obsy), c_x(cx), c_y(cy) {}

    template <typename T>
    bool operator()(const T* const pose, // SIMPLE_PINHOLE [r1,r2,r3,t1,t2,t3]
                    const T* const focal,  // shared_focal_length f
                    const T* const point,  // mapppoint coordination [X, Y, Z]
                    T* residuals) const {

        T p[3];

        ceres::AngleAxisRotatePoint(pose, point, p);

        p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T pred_u = focal[0] * xp + T(c_x);
        T pred_v = focal[0] * yp + T(c_y);

        residuals[0] = pred_u - T(obs_x);
        residuals[1] = pred_v - T(obs_y);
        return true;
    }

    double obs_x, obs_y, c_x, c_y;
};


void solve_ba_shared_focal(
    py::array_t<double> poses,              // (N, 6) - [r1,r2,r3, t1,t2,t3]
    py::array_t<double> points,             // (M, 3) - [X,Y,Z]
    py::array_t<double> focal,              // (1,)   - f
    py::array_t<double> observations,       // (K, 2) - [u, v]
    py::array_t<int> camera_indices,        // (K,)   - 每条观测对应的相机 ID
    py::array_t<int> point_indices,         // (K,)   - 每条观测对应的点 ID
    py::array_t<int> fixed_camera_indices,  // (k,)   - 不需要优化参数的相机 ID
    bool is_fixed_focal,                    // 是否固定焦距
    double c_x, double c_y                  // 固定主点坐标
) {
    // 1. 获取原始指针 (直接操作内存，实现零拷贝)
    auto p_poses = poses.mutable_data();
    auto p_points = points.mutable_data();
    auto p_focal = focal.mutable_data();
    auto p_obs = observations.data();
    auto p_camera_idx = camera_indices.data();
    auto p_point_idx = point_indices.data();
    auto p_fixed_camera_idx = fixed_camera_indices.data();

    // 2. 实例化 Ceres 问题
    ceres::Problem problem;

    // 鲁棒核函数：如果投影误差超过 1 像素，它会降低权重
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    // 3. 核心循环：将所有观测添加为残差块
    for (int i = 0; i < camera_indices.size(); ++i) {
        // 创建代价函数 (自动求导模式)
        // 参数含义: <结构体, 残差维度2, 相机参数维度6, 焦距参数维度1, 点参数维度3>
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<SnavelySharedFocalError, 2, 6, 1, 3>(
                new SnavelySharedFocalError(p_obs[2*i], p_obs[2*i+1], c_x, c_y));

        // 获取当前观测对应的相机和点在内存里的位置
        double* current_camera = p_poses + p_camera_idx[i] * 6;
        double* current_point = p_points + p_point_idx[i] *  3;

        problem.AddResidualBlock(cost_function, loss_function, current_camera, p_focal, current_point);
    }
    
    for (int i = 0; i < fixed_camera_indices.size(); ++i){
        int fixed_camera_idx = p_fixed_camera_idx[i];
        if (fixed_camera_idx >= 0 && fixed_camera_idx < poses.shape(0)){
            double* fixed_cam_ptr = p_poses + fixed_camera_idx * 6;
            problem.SetParameterBlockConstant(fixed_cam_ptr);
        }
    }

    if (is_fixed_focal){
        problem.SetParameterBlockConstant(p_focal);
    }

    // 4. 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; 
    options.minimizer_progress_to_stdout = true;    
    options.max_num_iterations = 100;                

    // 5. 启动优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl; 
    if (summary.termination_type == ceres::FAILURE) std::cout << "BA Optimization Failed!" << std::endl;
}

PYBIND11_MODULE(ba_core, m) {
    m.def("solve_ba_shared_focal", &solve_ba_shared_focal, "Solve BA with shared focal",
          py::arg("poses"), py::arg("points"), py::arg("focal"),
          py::arg("observations"), py::arg("camera_indices"), py::arg("point_indices"),
          py::arg("fixed_camera_indices"), py::arg("is_fixed_focal"),
          py::arg("c_x"), py::arg("c_y"));
}

