cmake_minimum_required(VERSION 3.16)
project(vio_calib_fg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(calib_core_lib OBJECT cpp/calib_core/calib.cpp)
# The object library is linked into both executables and a shared Python
# module; position-independent code is required for the latter on x86-64.
set_target_properties(calib_core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(calib_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(calib_core_lib PUBLIC Eigen3::Eigen)
target_compile_options(calib_core_lib PRIVATE -Wall -Wextra -O3)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(calib_core MODULE cpp/bindings.cpp)
  target_link_libraries(calib_core PRIVATE calib_core_lib)
  target_include_directories(calib_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
else()
  message(WARNING "pybind11 not found (pip install pybind11): Python module target skipped")
endif()

enable_testing()
add_executable(calib_core_test cpp/tests/calib_core_test.cpp)
target_link_libraries(calib_core_test PRIVATE calib_core_lib)
add_test(NAME calib_core_test COMMAND calib_core_test)

# Python<->C++ numeric alignment also works without pybind11: Python writes a
# problem file, this executable prints JSON, the Python test compares.
add_executable(alignment_dump cpp/tests/alignment_dump.cpp)
target_link_libraries(alignment_dump PRIVATE calib_core_lib)

# Optional OpenCV front end (undistort / KLT / PnP).  PCL is only checked, not
# required: no PCL algorithm is part of this calibration core.
find_package(OpenCV QUIET)
if(OpenCV_FOUND AND pybind11_FOUND)
  pybind11_add_module(calib_vision MODULE cpp/vision/vision.cpp)
  target_link_libraries(calib_vision PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
  target_include_directories(calib_vision PRIVATE ${OpenCV_INCLUDE_DIRS}
                             ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
  message(STATUS "OpenCV ${OpenCV_VERSION}: calib_vision target enabled")
elseif(OpenCV_FOUND)
  message(STATUS "OpenCV found but pybind11 missing: calib_vision target skipped")
else()
  message(STATUS "OpenCV not found: calib_vision target skipped")
endif()

find_package(PCL QUIET)
if(PCL_FOUND)
  message(STATUS "PCL found (${PCL_VERSION}); no PCL algorithm is built")
else()
  message(STATUS "PCL not found: PCL check skipped")
endif()
