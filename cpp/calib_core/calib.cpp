#include "calib.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace calib {

namespace {

// np.interp semantics: clamp outside the sample range.
double interp(double x, const std::vector<double>& xp,
              const Eigen::Matrix<double, Eigen::Dynamic, 3>& fp, int column) {
    if (x <= xp.front()) return fp(0, column);
    if (x >= xp.back()) return fp(xp.size() - 1, column);
    const auto it = std::upper_bound(xp.begin(), xp.end(), x);
    const std::size_t hi = static_cast<std::size_t>(it - xp.begin());
    const std::size_t lo = hi - 1;
    const double t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp(lo, column) + t * (fp(hi, column) - fp(lo, column));
}

Mat3 product_of_exponentials(const Eigen::Matrix<double, Eigen::Dynamic, 3>& inc,
                             std::vector<Mat3>* factors) {
    Mat3 R = Mat3::Identity();
    if (factors) factors->reserve(inc.rows());
    for (int i = 0; i < inc.rows(); ++i) {
        const Mat3 E = so3_exp(inc.row(i).transpose());
        R = R * E;
        if (factors) factors->push_back(E);
    }
    return R;
}

}  // namespace

double chi2_threshold(double level) {
    if (std::abs(level - 0.90) < 1e-9) return 6.2514;
    if (std::abs(level - 0.95) < 1e-9) return 7.8147;
    if (std::abs(level - 0.99) < 1e-9) return 11.3449;
    throw std::invalid_argument("chi2_level must be one of 0.90, 0.95, 0.99");
}

GyroTerms gyro_terms(double t0, double t1, const std::vector<double>& t_imu,
                     const Eigen::Matrix<double, Eigen::Dynamic, 3>& omega,
                     double dt_offset, const Vec3& bias) {
    GyroTerms terms;
    terms.a = t0 - dt_offset;
    terms.b = t1 - dt_offset;
    if (terms.b <= terms.a || t_imu.empty()) {
        terms.rates.resize(0, 3);
        terms.increments.resize(0, 3);
        terms.durations.resize(0);
        return terms;
    }
    const auto i0 = std::upper_bound(t_imu.begin(), t_imu.end(), terms.a);
    const auto i1 = std::lower_bound(t_imu.begin(), t_imu.end(), terms.b);
    std::vector<double> knots;
    knots.push_back(terms.a);
    knots.insert(knots.end(), i0, i1);
    knots.push_back(terms.b);

    terms.rates.resize(static_cast<int>(knots.size()), 3);
    for (std::size_t i = 0; i < knots.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            terms.rates(static_cast<int>(i), j) = interp(knots[i], t_imu, omega, j);
        }
        terms.rates.row(static_cast<int>(i)) -= bias.transpose();
    }
    terms.durations.resize(static_cast<int>(knots.size()) - 1);
    terms.increments.resize(static_cast<int>(knots.size()) - 1, 3);
    for (int i = 0; i + 1 < static_cast<int>(knots.size()); ++i) {
        terms.durations(i) = knots[i + 1] - knots[i];
        terms.increments.row(i) =
            0.5 * (terms.rates.row(i) + terms.rates.row(i + 1)) * terms.durations(i);
    }
    return terms;
}

Mat3 integrate_gyro(double t0, double t1, const std::vector<double>& t_imu,
                    const Eigen::Matrix<double, Eigen::Dynamic, 3>& omega,
                    double dt_offset, const Vec3& bias) {
    const GyroTerms terms = gyro_terms(t0, t1, t_imu, omega, dt_offset, bias);
    return product_of_exponentials(terms.increments, nullptr);
}

Eigen::VectorXd robust_weights(const Eigen::VectorXd& r, double huber_k,
                               double chi2_thresh) {
    const int M = static_cast<int>(r.size()) / 3;
    Eigen::VectorXd w = Eigen::VectorXd::Ones(M);
    for (int k = 0; k < M; ++k) {
        const double rho = r.segment<3>(3 * k).norm();
        if (rho * rho > chi2_thresh) {
            w(k) = 0.0;
        } else if (rho > huber_k) {
            w(k) = huber_k / rho;
        }
    }
    return w;
}

Eigen::VectorXd residual(const Eigen::VectorXd& theta,
                         const std::vector<Observation>& observations,
                         double sigma_r, bool estimate_bias) {
    const int n = estimate_bias ? 7 : 4;
    if (theta.size() != n) throw std::invalid_argument("theta has wrong size");
    const int M = static_cast<int>(observations.size());
    Eigen::VectorXd r = Eigen::VectorXd::Zero(3 * M);
    const Mat3 R_bc = so3_exp(theta.head<3>());
    Vec3 bias = Vec3::Zero();
    if (estimate_bias) bias = theta.segment<3>(4);
    for (int k = 0; k < M; ++k) {
        const Observation& o = observations[k];
        const Mat3 R_pred = R_bc * o.R_vis * R_bc.transpose();
        const GyroTerms terms = gyro_terms(o.t_k, o.t_k1, o.t_imu, o.omega,
                                           theta(3), bias);
        const Mat3 R_gyro = product_of_exponentials(terms.increments, nullptr);
        r.segment<3>(3 * k) = so3_log(R_pred.transpose() * R_gyro) / sigma_r;
    }
    return r;
}

Eigen::MatrixXd residual_jacobian(const Eigen::VectorXd& theta,
                                  const std::vector<Observation>& observations,
                                  double sigma_r, bool estimate_bias) {
    const int n = estimate_bias ? 7 : 4;
    if (theta.size() != n) throw std::invalid_argument("theta has wrong size");
    const int M = static_cast<int>(observations.size());
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3 * M, n);
    if (M == 0) return J;

    const Vec3 phi = theta.head<3>();
    const Mat3 R_bc = so3_exp(phi);
    const Mat3 Jl_phi = left_jacobian(phi);
    const double dt = theta(3);
    Vec3 bias = Vec3::Zero();
    if (estimate_bias) bias = theta.segment<3>(4);
    const double h = 1e-4;

    for (int k = 0; k < M; ++k) {
        const Observation& o = observations[k];
        const Mat3 R_pred = R_bc * o.R_vis * R_bc.transpose();
        const GyroTerms terms = gyro_terms(o.t_k, o.t_k1, o.t_imu, o.omega, dt, bias);
        std::vector<Mat3> factors;
        const Mat3 R_gyro = product_of_exponentials(terms.increments, &factors);
        const Vec3 r_k = so3_log(R_pred.transpose() * R_gyro);
        const Mat3 Jr_inv = right_jacobian_inverse(r_k);

        J.block<3, 3>(3 * k, 0) = Jr_inv
            * (R_gyro.transpose() * (R_pred - Mat3::Identity()) * Jl_phi) / sigma_r;

        const Mat3 R_plus = integrate_gyro(o.t_k, o.t_k1, o.t_imu, o.omega,
                                           dt + h, bias);
        const Mat3 R_minus = integrate_gyro(o.t_k, o.t_k1, o.t_imu, o.omega,
                                            dt - h, bias);
        J.block<3, 1>(3 * k, 3) =
            (so3_log(R_pred.transpose() * R_plus)
             - so3_log(R_pred.transpose() * R_minus)) / (2.0 * h * sigma_r);

        if (estimate_bias && terms.increments.rows() > 0) {
            std::vector<Mat3> dR(3, Mat3::Zero());
            Mat3 Ri = Mat3::Identity();
            for (int i = 0; i < terms.increments.rows(); ++i) {
                const Mat3& E = factors[i];
                const Vec3 inc = terms.increments.row(i).transpose();
                const Mat3 Jr_inc = left_jacobian(-inc);
                for (int j = 0; j < 3; ++j) {
                    const Vec3 dtheta = -terms.durations(i) * Vec3::Unit(j);
                    const Mat3 dE = E * hat(Jr_inc * dtheta);
                    dR[j] = dR[j] * E + Ri * dE;
                }
                Ri = Ri * E;
            }
            for (int j = 0; j < 3; ++j) {
                J.block<3, 1>(3 * k, 4 + j) =
                    Jr_inv * vee(R_gyro.transpose() * dR[j]) / sigma_r;
            }
        }
    }
    return J;
}

CalibResult batch_calibrate(const std::vector<Observation>& observations,
                            const Eigen::VectorXd& theta0, double sigma_r,
                            bool estimate_bias, int max_iters, double damping,
                            double huber_k, double chi2_level, double tol) {
    const double chi2 = chi2_threshold(chi2_level);
    Eigen::VectorXd theta = theta0;
    CalibResult result;
    Eigen::VectorXd w = Eigen::VectorXd::Ones(static_cast<int>(observations.size()));
    for (int iter = 0; iter < max_iters; ++iter) {
        const Eigen::VectorXd r = residual(theta, observations, sigma_r, estimate_bias);
        const Eigen::MatrixXd J =
            residual_jacobian(theta, observations, sigma_r, estimate_bias);
        w = robust_weights(r, huber_k, chi2);
        Eigen::VectorXd W(r.size());
        for (int i = 0; i < r.size(); ++i) W(i) = w(i / 3);
        const Eigen::MatrixXd H = J.transpose() * W.asDiagonal() * J
                                  + damping * Eigen::MatrixXd::Identity(theta.size(), theta.size());
        const Eigen::VectorXd g = J.transpose() * (W.array() * r.array()).matrix();
        const Eigen::VectorXd delta = H.ldlt().solve(-g);
        theta += delta;
        result.cost = 0.5 * (r.array() * W.array() * r.array()).sum();
        result.nfev = iter + 1;
        if (delta.norm() < tol) break;
    }
    result.rvec_bc = theta.head<3>();
    result.dt = theta(3);
    result.bias_g = Vec3::Zero();
    if (estimate_bias) result.bias_g = theta.segment<3>(4);
    result.n_outliers = static_cast<int>((w.array() == 0.0).count());
    return result;
}

}  // namespace calib
