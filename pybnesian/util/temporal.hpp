#ifndef PYBNESIAN_UTIL_TEMPORAL_HPP
#define PYBNESIAN_UTIL_TEMPORAL_HPP

#include <string>
#include <vector>

namespace util {

std::string temporal_name(const std::string& name, int slice_index);
std::vector<std::string> temporal_names(const std::vector<std::string>& variables,
                                        int offset_slice,
                                        int markovian_order);
std::vector<std::vector<std::string>> temporal_slice_names(const std::vector<std::string>& variables,
                                                           int start_slice,
                                                           int markovian_order);

}  // namespace util

#endif  // PYBNESIAN_UTIL_TEMPORAL_HPP