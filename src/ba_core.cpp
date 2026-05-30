#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <string>

namespace py = pybind11;

// Reprojection residual for a SIMPLE_PINHOLE camera with one shared focal.
struct SnavelySharedFocalError {

    SnavelySharedFocalError(double obsx, double obsy, double cx, double cy)
        : obs_x(obsx), obs_y(obsy), c_x(cx), c_y(cy) {}

    template <typename T>
    bool operator()(const T* const pose,   // [angle_axis(3), translation(3)]
                    const T* const focal,  // shared focal length
                    const T* const point,  // world point [X, Y, Z]
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


std::string solve_ba_shared_focal(
    py::array_t<double> poses,              // (N, 6)
    py::array_t<double> points,             // (M, 3)
    py::array_t<double> focal,              // (1,)
    py::array_t<double> observations,       // (K, 2)
    py::array_t<int> camera_indices,        // (K,)
    py::array_t<int> point_indices,         // (K,)
    py::array_t<int> fixed_camera_indices,  // (F,)
    bool is_fixed_focal,
    double c_x, double c_y
) {
    // Work directly on numpy buffers so Python sees optimized values.
    auto p_poses = poses.mutable_data();
    auto p_points = points.mutable_data();
    auto p_focal = focal.mutable_data();
    auto p_obs = observations.data();
    auto p_camera_idx = camera_indices.data();
    auto p_point_idx = point_indices.data();
    auto p_fixed_camera_idx = fixed_camera_indices.data();

    ceres::Problem problem;

    // Robust loss limits the influence of residual outliers.
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < camera_indices.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<SnavelySharedFocalError, 2, 6, 1, 3>(
                new SnavelySharedFocalError(p_obs[2*i], p_obs[2*i+1], c_x, c_y));

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

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; 
    options.minimizer_progress_to_stdout = false;    
    options.max_num_iterations = 100;                

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return summary.BriefReport();
}

PYBIND11_MODULE(ba_core, m) {
    m.def("solve_ba_shared_focal", &solve_ba_shared_focal, "Solve BA with shared focal",
          py::arg("poses"), py::arg("points"), py::arg("focal"),
          py::arg("observations"), py::arg("camera_indices"), py::arg("point_indices"),
          py::arg("fixed_camera_indices"), py::arg("is_fixed_focal"),
          py::arg("c_x"), py::arg("c_y"));
}

