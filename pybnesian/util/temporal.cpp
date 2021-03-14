#include <util/temporal.hpp>

namespace util {

std::string temporal_name(const std::string& name, int slice_index) {
    return name + "_t_" + std::to_string(slice_index);
}

std::vector<std::string> temporal_names(const std::vector<std::string>& variables,
                                        int offset_slice,
                                        int markovian_order) {
    int num_slices = markovian_order - offset_slice + 1;
    std::vector<std::string> names;
    names.reserve(variables.size() * num_slices);

    for (const auto& v : variables) {
        for (int i = offset_slice; i <= markovian_order; ++i) {
            names.push_back(temporal_name(v, i));
        }
    }

    return names;
}

std::vector<std::vector<std::string>> temporal_slice_names(const std::vector<std::string>& variables,
                                                           int start_slice,
                                                           int markovian_order) {
    std::vector<std::vector<std::string>> temporal_slice_names;
    temporal_slice_names.reserve(markovian_order - start_slice + 1);

    for (auto i = start_slice; i <= markovian_order; ++i) {
        std::vector<std::string> slice_names;
        slice_names.reserve(variables.size());

        for (const auto& var : variables) {
            slice_names.push_back(util::temporal_name(var, i));
        }

        temporal_slice_names.push_back(slice_names);
    }

    return temporal_slice_names;
}

}  // namespace util