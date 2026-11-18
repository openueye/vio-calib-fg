// pybind11 wrapper around calib_core.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "calib_core/calib.h"

namespace py = pybind11;
using namespace calib;

namespace {

std::vector<double> to_vector(const Eigen::VectorXd& values) {
    return std::vector<double>(values.data(), values.data() + values.size());
}

std::vector<Observation> to_observations(const py::list& items) {
    std::vector<Observation> observations;
    observations.reserve(items.size());
    for (const py::handle& item : items) {
        const py::tuple tuple = item.cast<py::tuple>();
        if (tuple.size() != 5) {
            throw std::invalid_argument(
                "observation must be (R_vis, t_k, t_k1, t_imu, omega)");
        }
        Observation observation;
        observation.R_vis = tuple[0].cast<Mat3>();
        observation.t_k = tuple[1].cast<double>();
        observation.t_k1 = tuple[2].cast<double>();
        const Eigen::VectorXd t_imu = tuple[3].cast<Eigen::VectorXd>();
        observation.t_imu = to_vector(t_imu);
        observation.omega = tuple[4].cast<Eigen::MatrixXd>();
        if (observation.omega.cols() != 3 ||
            observation.omega.rows() != static_cast<int>(observation.t_imu.size())) {
            throw std::invalid_argument("omega must be N x 3 matching t_imu");
        }
        observations.push_back(std::move(observation));
    }
    return observations;
}

Eigen::Matrix<double, Eigen::Dynamic, 3> to_omega(const Eigen::MatrixXd& omega) {
    if (omega.cols() != 3) throw std::invalid_argument("omega must be N x 3");
    return omega;
}

}  // namespace

PYBIND11_MODULE(calib_core, m) {
    m.doc() = "C++/Eigen core for online camera-IMU calibration (mirror of Python)";

    m.def("hat", [](const Vec3& v) { return hat(v); });
    m.def("vee", [](const Mat3& S) { return vee(S); });
    m.def("so3_exp", [](const Vec3& phi) { return so3_exp(phi); });
    m.def("so3_log", [](const Mat3& R) { return so3_log(R); });
    m.def("left_jacobian", [](const Vec3& phi) { return left_jacobian(phi); });
    m.def("left_jacobian_inverse",
          [](const Vec3& phi) { return left_jacobian_inverse(phi); });
    m.def("right_jacobian_inverse",
          [](const Vec3& phi) { return right_jacobian_inverse(phi); });
    m.def("rotation_error", [](const Mat3& R1, const Mat3& R2) {
        return rotation_error(R1, R2);
    });

    m.def("integrate_gyro",
          [](double t0, double t1, const Eigen::VectorXd& t_imu,
             const Eigen::MatrixXd& omega, double dt_offset, const Vec3& bias) {
              return integrate_gyro(t0, t1, to_vector(t_imu), to_omega(omega),
                                    dt_offset, bias);
          },
          py::arg("t0"), py::arg("t1"), py::arg("t_imu"), py::arg("omega"),
          py::arg("dt_offset") = 0.0, py::arg("bias") = Vec3::Zero());

    m.def("robust_weights", &robust_weights, py::arg("r"), py::arg("huber_k"),
          py::arg("chi2_thresh"));

    m.def("residual",
          [](const Eigen::VectorXd& theta, const py::list& observations,
             double sigma_r, bool estimate_bias) {
              return residual(theta, to_observations(observations), sigma_r,
                              estimate_bias);
          },
          py::arg("theta"), py::arg("observations"), py::arg("sigma_r") = 0.01,
          py::arg("estimate_bias") = false);

    m.def("residual_jacobian",
          [](const Eigen::VectorXd& theta, const py::list& observations,
             double sigma_r, bool estimate_bias) {
              const auto obs = to_observations(observations);
              return py::make_tuple(
                  residual(theta, obs, sigma_r, estimate_bias),
                  residual_jacobian(theta, obs, sigma_r, estimate_bias));
          },
          py::arg("theta"), py::arg("observations"), py::arg("sigma_r") = 0.01,
          py::arg("estimate_bias") = false);

    m.def("batch_calibrate",
          [](const py::list& observations, const Eigen::VectorXd& theta0,
             double sigma_r, bool estimate_bias, int max_iters, double damping,
             double huber_k, double chi2_level) {
              const CalibResult result = batch_calibrate(
                  to_observations(observations), theta0, sigma_r, estimate_bias,
                  max_iters, damping, huber_k, chi2_level);
              py::dict out;
              out["rvec_bc"] = result.rvec_bc;
              out["dt"] = result.dt;
              out["bias_g"] = result.bias_g;
              out["cost"] = result.cost;
              out["nfev"] = result.nfev;
              out["n_outliers"] = result.n_outliers;
              return out;
          },
          py::arg("observations"), py::arg("theta0"), py::arg("sigma_r") = 0.01,
          py::arg("estimate_bias") = false, py::arg("max_iters") = 20,
          py::arg("damping") = 1e-6, py::arg("huber_k") = 1.345,
          py::arg("chi2_level") = 0.99);
}
