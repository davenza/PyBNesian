#ifndef PYBNESIAN_UTIL_EXCEPTIONS_HPP
#define PYBNESIAN_UTIL_EXCEPTIONS_HPP

namespace util {

class singular_covariance_data : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

}  // namespace util

#endif  // PYBNESIAN_UTIL_EXCEPTIONS_HPP
