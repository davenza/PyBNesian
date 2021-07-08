#ifndef PYBNESIAN_UTIL_VECH_OPS_HPP
#define PYBNESIAN_UTIL_VECH_OPS_HPP

#include <Eigen/Dense>

using Eigen::VectorXd, Eigen::MatrixXd;

namespace util {

VectorXd vech(const MatrixXd& m);
MatrixXd invvech(const VectorXd& m);
MatrixXd invvech_triangular(const VectorXd& v);

}  // namespace util

#endif  // PYBNESIAN_VECH_OPS_HPP