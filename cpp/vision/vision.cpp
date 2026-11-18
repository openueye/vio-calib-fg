// Optional OpenCV front-end helpers exposed to Python for numeric checks.
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace {

cv::Mat to_gray(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image) {
    const py::buffer_info info = image.request();
    if (info.ndim != 2) throw std::invalid_argument("image must be 2-D uint8");
    return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                   CV_8UC1, info.ptr);
}

}  // namespace

PYBIND11_MODULE(calib_vision, m) {
    m.doc() = "Optional OpenCV front end (undistort, KLT, PnP RANSAC)";

    m.def("undistort_points",
          [](const Eigen::MatrixXd& points, const Eigen::Matrix3d& K,
             const std::vector<double>& dist) {
              std::vector<cv::Point2d> source;
              source.reserve(points.rows());
              for (int i = 0; i < points.rows(); ++i) {
                  source.emplace_back(points(i, 0), points(i, 1));
              }
              cv::Mat camera = (cv::Mat_<double>(3, 3) << K(0, 0), K(0, 1), K(0, 2),
                                K(1, 0), K(1, 1), K(1, 2), K(2, 0), K(2, 1), K(2, 2));
              cv::Mat distortion(1, static_cast<int>(dist.size()), CV_64F,
                                 const_cast<double*>(dist.data()));
              std::vector<cv::Point2d> normalised;
              cv::undistortPoints(source, normalised, camera, distortion);
              Eigen::MatrixXd out(normalised.size(), 2);
              for (std::size_t i = 0; i < normalised.size(); ++i) {
                  out(static_cast<int>(i), 0) = normalised[i].x;
                  out(static_cast<int>(i), 1) = normalised[i].y;
              }
              return out;
          },
          py::arg("points"), py::arg("K"), py::arg("dist"));

    m.def("track_klt",
          [](const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& previous,
             const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& next,
             const Eigen::MatrixXd& points, int window, int levels) {
              const cv::Mat prev = to_gray(previous);
              const cv::Mat curr = to_gray(next);
              std::vector<cv::Point2f> source, tracked;
              source.reserve(points.rows());
              for (int i = 0; i < points.rows(); ++i) {
                  source.emplace_back(static_cast<float>(points(i, 0)),
                                      static_cast<float>(points(i, 1)));
              }
              std::vector<unsigned char> status;
              std::vector<float> error;
              cv::calcOpticalFlowPyrLK(prev, curr, source, tracked, status, error,
                                       cv::Size(window, window), levels);
              Eigen::MatrixXd out(static_cast<int>(tracked.size()), 2);
              std::vector<int> ok(tracked.size());
              for (std::size_t i = 0; i < tracked.size(); ++i) {
                  out(static_cast<int>(i), 0) = tracked[i].x;
                  out(static_cast<int>(i), 1) = tracked[i].y;
                  ok[i] = status[i] ? 1 : 0;
              }
              return py::make_tuple(out, ok);
          },
          py::arg("previous"), py::arg("next"), py::arg("points"),
          py::arg("window") = 21, py::arg("levels") = 3);

    m.def("solve_pnp",
          [](const Eigen::MatrixXd& object_points, const Eigen::MatrixXd& image_points,
             const Eigen::Matrix3d& K, const std::vector<double>& dist) {
              if (object_points.rows() != image_points.rows()) {
                  throw std::invalid_argument("point counts differ");
              }
              std::vector<cv::Point3f> object;
              std::vector<cv::Point2f> image;
              for (int i = 0; i < object_points.rows(); ++i) {
                  object.emplace_back(static_cast<float>(object_points(i, 0)),
                                      static_cast<float>(object_points(i, 1)),
                                      static_cast<float>(object_points(i, 2)));
                  image.emplace_back(static_cast<float>(image_points(i, 0)),
                                     static_cast<float>(image_points(i, 1)));
              }
              cv::Mat camera = (cv::Mat_<double>(3, 3) << K(0, 0), K(0, 1), K(0, 2),
                                K(1, 0), K(1, 1), K(1, 2), K(2, 0), K(2, 1), K(2, 2));
              cv::Mat distortion(1, static_cast<int>(dist.size()), CV_64F,
                                 const_cast<double*>(dist.data()));
              cv::Vec3d rvec, tvec;
              std::vector<int> inliers;
              const bool ok = cv::solvePnPRansac(
                  object, image, camera, distortion, rvec, tvec, false, 100,
                  3.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
              if (!ok) throw std::runtime_error("solvePnPRansac failed");
              py::dict out;
              out["rvec"] = Eigen::Vector3d(rvec[0], rvec[1], rvec[2]);
              out["tvec"] = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
              out["inliers"] = static_cast<int>(inliers.size());
              return out;
          },
          py::arg("object_points"), py::arg("image_points"), py::arg("K"),
          py::arg("dist"));
}
