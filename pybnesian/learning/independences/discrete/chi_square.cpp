#include <learning/independences/discrete/chi_square.hpp>
#include <factors/discrete/discrete_indices.hpp>
#include <util/math_constants.hpp>
#include <boost/math/distributions/chi_squared.hpp>

namespace learning::independences::discrete {

double ChiSquare::pvalue(const std::string& v1, const std::string& v2) const {
    std::vector<std::string> dummy_v2{v2};
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, v1, dummy_v2);
    auto joint_counts = factors::discrete::joint_counts(m_df, v1, dummy_v2, cardinality, strides);

    auto v1_marg = factors::discrete::marginal_counts(joint_counts, 0, cardinality, strides);
    auto v2_marg = factors::discrete::marginal_counts(joint_counts, 1, cardinality, strides);

    auto inv_obs = 1. / joint_counts.sum();

    double statistic = 0;
    for (int i = 0; i < cardinality(0); ++i) {
        for (int j = 0; j < cardinality(1); ++j) {
            auto expected = static_cast<double>(v1_marg(i) * v2_marg(j)) * inv_obs;

            if (expected != 0) {
                auto index = i + j * strides(1);

                auto d = joint_counts(index) - expected;
                statistic += d * d / expected;
            }
        }
    }

    auto df = (cardinality(0) - 1) * (cardinality(1) - 1);

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, statistic));
}

double ChiSquare::pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const {
    std::vector<std::string> dummy_vars{v2, ev};
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, v1, dummy_vars);
    auto joint_counts = factors::discrete::joint_counts(m_df, v1, dummy_vars, cardinality, strides);

    auto evidence_marg = factors::discrete::marginal_counts(joint_counts, 2, cardinality, strides);

    auto evidence_configurations = cardinality(2);
    auto vars_configurations = strides(2);

    double statistic = 0;

    for (auto k = 0; k < evidence_configurations; ++k) {
        if (evidence_marg(k) == 0) continue;

        auto offset = k * vars_configurations;
        auto evidence_segment = joint_counts.segment(offset, vars_configurations);

        auto v1_marg = factors::discrete::marginal_counts(evidence_segment, 0, cardinality, strides);
        auto v2_marg = factors::discrete::marginal_counts(evidence_segment, 1, cardinality, strides);

        auto inv_obs = 1. / evidence_marg(k);

        for (int i = 0; i < cardinality(0); ++i) {
            for (int j = 0; j < cardinality(1); ++j) {
                auto expected = static_cast<double>(v1_marg(i) * v2_marg(j)) * inv_obs;

                if (expected != 0) {
                    auto index = offset + i + j * strides(1);

                    auto d = joint_counts(index) - expected;
                    statistic += d * d / expected;
                }
            }
        }
    }

    auto df = (cardinality(0) - 1) * (cardinality(1) - 1) * cardinality(2);

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, statistic));
}

double ChiSquare::pvalue(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const {
    std::vector<std::string> dummy_vars{v2};
    dummy_vars.reserve(ev.size() + 1);
    dummy_vars.insert(dummy_vars.end(), ev.begin(), ev.end());

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, v1, dummy_vars);
    auto joint_counts = factors::discrete::joint_counts(m_df, v1, dummy_vars, cardinality, strides);

    auto evidence_configurations = cardinality.tail(ev.size()).prod();
    auto vars_configurations = cardinality(0) * cardinality(1);

    double statistic = 0;

    for (auto k = 0; k < evidence_configurations; ++k) {
        auto offset = k * vars_configurations;

        int total_sum = 0;
        auto marginal_v1 = VectorXi::Zero(cardinality(0)).eval();
        auto marginal_v2 = VectorXi::Zero(cardinality(1)).eval();

        for (auto i = 0; i < cardinality(0); ++i) {
            for (auto j = 0; j < cardinality(1); ++j) {
                auto c = joint_counts(offset + i + j * strides(1));
                marginal_v1(i) += c;
                marginal_v2(j) += c;
                total_sum += c;
            }
        }

        if (total_sum == 0) continue;

        auto inv_obs = 1. / static_cast<double>(total_sum);

        for (auto i = 0; i < cardinality(0); ++i) {
            for (auto j = 0; j < cardinality(1); ++j) {
                auto expected = static_cast<double>(marginal_v1(i) * marginal_v2(j)) * inv_obs;

                if (expected != 0) {
                    auto c = joint_counts(offset + i + j * strides(1));
                    auto d = c - expected;

                    statistic += d * d / expected;
                }
            }
        }
    }

    // Avoids error: OverflowError: Error in function boost::math::tgamma<long double>(long double): Result of tgamma is
    // too large to represent. of Boost, when statistic is very close to 0.
    if (statistic < util::machine_tol) {
        return 1;
    }

    auto df = (cardinality(0) - 1) * (cardinality(1) - 1) * evidence_configurations;

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, statistic));
}

}  // namespace learning::independences::discrete