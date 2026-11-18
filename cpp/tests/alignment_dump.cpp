// Standalone Python<->C++ alignment harness (no pybind11 required).
//
// Reads a plain-text problem, prints JSON.  The Python test
// tests/test_cpp_alignment_dump.py writes the problem, runs this executable and
// compares the residual, Jacobian, robust weights and batch result.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "calib_core/calib.h"

using namespace calib;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: alignment_dump <problem.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }
    int M = 0, n = 0, estimate_bias = 0;
    double sigma = 0.01;
    in >> M >> n >> sigma >> estimate_bias;
    std::vector<Observation> observations;
    for (int k = 0; k < M; ++k) {
        Observation o;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) in >> o.R_vis(r, c);
        }
        in >> o.t_k >> o.t_k1;
        int N = 0;
        in >> N;
        o.t_imu.resize(N);
        for (int i = 0; i < N; ++i) in >> o.t_imu[i];
        o.omega.resize(N, 3);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 3; ++j) in >> o.omega(i, j);
        }
        observations.push_back(std::move(o));
    }
    Eigen::VectorXd theta(n);
    for (int i = 0; i < n; ++i) in >> theta(i);
    int max_iters = 20;
    double damping = 1e-6, huber = 1.345, chi2 = 0.99;
    in >> max_iters >> damping >> huber >> chi2;

    const Eigen::VectorXd r = residual(theta, observations, sigma, estimate_bias != 0);
    const Eigen::MatrixXd J =
        residual_jacobian(theta, observations, sigma, estimate_bias != 0);
    const Eigen::VectorXd w = robust_weights(r, huber, chi2_threshold(chi2));
    const CalibResult batch =
        batch_calibrate(observations, theta, sigma, estimate_bias != 0, max_iters,
                        damping, huber, chi2);

    std::cout << std::setprecision(17);
    std::cout << "{\"r\":[";
    for (int i = 0; i < r.size(); ++i) std::cout << (i ? "," : "") << r(i);
    std::cout << "],\"J\":[";
    bool first = true;
    for (int i = 0; i < J.rows(); ++i) {
        for (int j = 0; j < J.cols(); ++j) {
            std::cout << (first ? "" : ",") << J(i, j);
            first = false;
        }
    }
    std::cout << "],\"w\":[";
    for (int i = 0; i < w.size(); ++i) std::cout << (i ? "," : "") << w(i);
    std::cout << "],\"batch\":{\"rvec_bc\":[";
    for (int i = 0; i < 3; ++i) std::cout << (i ? "," : "") << batch.rvec_bc(i);
    std::cout << "],\"dt\":" << batch.dt << ",\"bias_g\":[";
    for (int i = 0; i < 3; ++i) std::cout << (i ? "," : "") << batch.bias_g(i);
    std::cout << "],\"cost\":" << batch.cost << ",\"nfev\":" << batch.nfev
              << ",\"n_outliers\":" << batch.n_outliers << "}}\n";
    return 0;
}
