// Self-contained numeric checks for the C++ core (no Python needed).
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "calib_core/calib.h"

using namespace calib;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

void check_close(double actual, double expected, double tolerance,
                 const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " (" << actual << " vs " << expected
                  << ")\n";
        ++failures;
    }
}

}  // namespace

int main() {
    const Vec3 phi(0.03, -0.02, 0.05);
    const Mat3 R = so3_exp(phi);
    check_close((so3_log(R) - phi).norm(), 0.0, 1e-12, "exp/log roundtrip");
    check_close(R.determinant(), 1.0, 1e-12, "rotation determinant");
    check_close((R * R.transpose() - Mat3::Identity()).norm(), 0.0, 1e-12,
                "rotation orthogonality");
    check_close((left_jacobian(phi) * left_jacobian_inverse(phi)
                 - Mat3::Identity()).norm(), 0.0, 1e-10, "left Jacobian inverse");
    check_close((hat(phi) + hat(phi).transpose()).norm(), 0.0, 1e-15,
                "hat skew symmetry");
    check_close((vee(hat(phi)) - phi).norm(), 0.0, 1e-15, "hat/vee roundtrip");

    // Constant body rate: integrate exactly.
    std::vector<double> t_imu;
    for (int i = 0; i < 400; ++i) t_imu.push_back(0.005 * i);
    Eigen::Matrix<double, Eigen::Dynamic, 3> omega(400, 3);
    omega.setConstant(0.0);
    omega.col(2).setConstant(0.53 + 0.03);
    const Vec3 bias(0.0, 0.0, 0.03);
    const Mat3 integrated = integrate_gyro(0.3, 0.7, t_imu, omega, 0.0, bias);
    const Mat3 expected = so3_exp(Vec3(0.0, 0.0, 0.53 * 0.4));
    check_close((integrated - expected).norm(), 0.0, 1e-12,
                "integrate constant rate");

    // A residual built from a consistent visual rotation is zero.
    Observation observation;
    const Vec3 rate(0.1, -0.2, 0.7);
    const double interval = 0.2;
    const Mat3 body_relative = so3_exp(rate * interval);
    observation.R_vis = R.transpose() * body_relative * R;
    observation.t_k = 0.0;
    observation.t_k1 = interval;
    observation.t_imu = t_imu;
    observation.omega.resize(400, 3);
    for (int i = 0; i < 400; ++i) observation.omega.row(i) = rate.transpose();

    Eigen::VectorXd theta(4);
    theta.head<3>() = phi;
    theta(3) = 0.0;
    const std::vector<Observation> observations{observation};
    const Eigen::VectorXd r = residual(theta, observations, 0.01, false);
    check_close(r.norm(), 0.0, 1e-12, "zero residual at truth");

    // The analytic Jacobian must match a central difference.
    theta(3) = 0.01;
    const Eigen::MatrixXd J = residual_jacobian(theta, observations, 0.01, false);
    const double step = 1e-6;
    for (int j = 0; j < 4; ++j) {
        Eigen::VectorXd plus = theta, minus = theta;
        plus(j) += step;
        minus(j) -= step;
        const Eigen::VectorXd numeric =
            (residual(plus, observations, 0.01, false)
             - residual(minus, observations, 0.01, false)) / (2.0 * step);
        check_close((J.col(j) - numeric).norm(), 0.0, 2e-4,
                    "Jacobian column versus finite difference");
    }

    // Robust weights: inside, Huber zone, rejected.
    Eigen::VectorXd residuals(9);
    residuals << 0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 5.0, 5.0, 5.0;
    const Eigen::VectorXd w = robust_weights(residuals, 1.345, 11.3449);
    check(w(0) == 1.0, "Huber inside weight");
    check(w(1) > 0.0 && w(1) < 1.0, "Huber downweight");
    check(w(2) == 0.0, "chi-square rejection");

    if (failures == 0) {
        std::cout << "calib_core_test: all checks passed\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
}
