// C++/Eigen mirror of src/online_calib.py.
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <vector>

#include "so3.h"

namespace calib {

struct Observation {
    Mat3 R_vis;
    double t_k = 0.0;
    double t_k1 = 0.0;
    std::vector<double> t_imu;
    Eigen::Matrix<double, Eigen::Dynamic, 3> omega;
};

struct GyroTerms {
    double a = 0.0;
    double b = 0.0;
    Eigen::Matrix<double, Eigen::Dynamic, 3> rates;       // bias-corrected
    Eigen::Matrix<double, Eigen::Dynamic, 3> increments;
    Eigen::VectorXd durations;
};

struct CalibResult {
    Vec3 rvec_bc = Vec3::Zero();
    double dt = 0.0;
    Vec3 bias_g = Vec3::Zero();
    double cost = 0.0;
    int nfev = 0;
    int n_outliers = 0;
};

// Chi-square threshold for 3 degrees of freedom (matches CHI2_3 in Python).
double chi2_threshold(double level);

GyroTerms gyro_terms(double t0, double t1, const std::vector<double>& t_imu,
                     const Eigen::Matrix<double, Eigen::Dynamic, 3>& omega,
                     double dt_offset, const Vec3& bias);

Mat3 integrate_gyro(double t0, double t1, const std::vector<double>& t_imu,
                    const Eigen::Matrix<double, Eigen::Dynamic, 3>& omega,
                    double dt_offset, const Vec3& bias);

Eigen::VectorXd robust_weights(const Eigen::VectorXd& r, double huber_k,
                               double chi2_thresh);

Eigen::VectorXd residual(const Eigen::VectorXd& theta,
                         const std::vector<Observation>& observations,
                         double sigma_r, bool estimate_bias);

Eigen::MatrixXd residual_jacobian(const Eigen::VectorXd& theta,
                                  const std::vector<Observation>& observations,
                                  double sigma_r, bool estimate_bias);

CalibResult batch_calibrate(const std::vector<Observation>& observations,
                            const Eigen::VectorXd& theta0, double sigma_r,
                            bool estimate_bias, int max_iters, double damping,
                            double huber_k, double chi2_level,
                            double tol = 1e-10);

}  // namespace calib
