#include <learning/independences/independence.hpp>

namespace learning::independences {

    std::vector<std::vector<std::string>> temporal_slice_names(const std::vector<std::string>& variables,
                                                               int start_slice,
                                                               int markovian_order) {
        std::vector<std::vector<std::string>> temporal_slice_names;
        temporal_slice_names.reserve(markovian_order - start_slice + 1);

        for (auto i = start_slice; i <= markovian_order; ++i) {
            std::vector<std::string> slice_names;
            slice_names.reserve(variables.size());

            for (const auto& var : variables) {
                slice_names.push_back(dataset::temporal_name(var, i));
            }

            temporal_slice_names.push_back(slice_names);
        }

        return temporal_slice_names;
    }
    
    ArcStringVector DynamicIndependenceTest::static_blacklist() const {
        auto markovian = markovian_order();
        if (markovian == 1)
            return ArcStringVector();

        ArcStringVector blacklist;
        blacklist.reserve(num_variables() * num_variables() * markovian * (markovian - 1) / 2);

        auto variables = variable_names();
        auto slice_names = temporal_slice_names(variables, 1, markovian);

        for (int i = 0, end = markovian - 1; i < end; ++i) {
            for (const auto& source : slice_names[i]) {
                for (auto j = i + 1; j < markovian; ++j) {
                    for (const auto& dest : slice_names[j]) {
                        blacklist.push_back(std::make_pair(source, dest));
                    }
                }
            }
        }

        return blacklist;
    };

    ArcStringVector DynamicIndependenceTest::transition_blacklist() const {
        auto markovian = markovian_order();
        
        ArcStringVector blacklist;
        blacklist.reserve(num_variables() * num_variables() * markovian * (markovian + 1) / 2);

        auto variables = variable_names();
        auto slice_names = temporal_slice_names(variables, 0, markovian);

        for (int i = 0; i < markovian; ++i) {
            for (const auto& source : slice_names[i]) {
                for (auto j = i + 1; j <= markovian; ++j) {
                    for (const auto& dest : slice_names[j]) {
                        blacklist.push_back(std::make_pair(source, dest));
                    }
                }
            }
        }

        return blacklist;
    };
}