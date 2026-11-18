// Self-contained SO(3) math mirrored from src/so3.py.
//
// The formulas (including the small-angle branches) are identical to the
// NumPy reference so the two implementations agree to numerical precision.
#pragma once

#include <Eigen/Core>
#include <cmath>

namespace calib {

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

inline Mat3 hat(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return S;
}

inline Vec3 vee(const Mat3& S) {
    return Vec3(S(2, 1), S(0, 2), S(1, 0));
}

inline Mat3 so3_exp(const Vec3& phi) {
    const double theta = phi.norm();
    const Mat3 K = hat(phi);
    if (theta < 1e-12) {
        return Mat3::Identity() + K;
    }
    const Vec3 axis = phi / theta;
    const Mat3 Kaxis = hat(axis);
    return Mat3::Identity() + std::sin(theta) * Kaxis
           + (1.0 - std::cos(theta)) * (Kaxis * Kaxis);
}

inline Vec3 so3_log(const Mat3& R) {
    double c = 0.5 * (R.trace() - 1.0);
    c = std::min(1.0, std::max(-1.0, c));
    const double theta = std::acos(c);
    const Mat3 S = 0.5 * (R - R.transpose());
    const Vec3 v = vee(S);
    if (theta < 1e-12) {
        return v;
    }
    return (theta / std::sin(theta)) * v;
}

inline Mat3 left_jacobian(const Vec3& phi) {
    const double theta = phi.norm();
    const Mat3 K = hat(phi);
    if (theta < 1e-12) {
        return Mat3::Identity() + 0.5 * K;
    }
    return Mat3::Identity()
           + (1.0 - std::cos(theta)) / (theta * theta) * K
           + (theta - std::sin(theta)) / (theta * theta * theta) * (K * K);
}

inline Mat3 left_jacobian_inverse(const Vec3& phi) {
    const double theta = phi.norm();
    const Mat3 K = hat(phi);
    if (theta < 1e-12) {
        return Mat3::Identity() - 0.5 * K;
    }
    const double a = 1.0 / (theta * theta);
    const double b = (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    return Mat3::Identity() - 0.5 * K + (a - b) * (K * K);
}

inline Mat3 right_jacobian_inverse(const Vec3& phi) {
    return left_jacobian_inverse(-phi);
}

inline double rotation_error(const Mat3& R1, const Mat3& R2) {
    return so3_log(R1.transpose() * R2).norm();
}

}  // namespace calib
