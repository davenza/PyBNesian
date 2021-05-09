#include <factors/continuous/LSCDE.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

namespace factors::continuous {

std::shared_ptr<Factor> LSCDEType::new_factor(const BayesianNetworkBase&,
                                              const std::string& variable,
                                              const std::vector<std::string>& evidence) const {
    return std::make_shared<LSCDE>(variable, evidence);
}

std::shared_ptr<Factor> LSCDEType::new_factor(const ConditionalBayesianNetworkBase&,
                                              const std::string& variable,
                                              const std::vector<std::string>& evidence) const {
    return std::make_shared<LSCDE>(variable, evidence);
}

std::shared_ptr<FactorType> LSCDEType::opposite_semiparametric() const { return LinearGaussianCPDType::get(); }

// https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
std::unordered_set<int> indices_without_replacement(int N, int k, std::mt19937& gen) {
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(0, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

std::pair<DataFrame, DataFrame> LSCDE::get_uv(const DataFrame& df) {
    bool contains_null = df.null_count(variable(), evidence()) > 0;

    if (!contains_null && df->num_rows() <= m_b) return std::make_pair(df.loc(evidence()), df.loc(variable()));

    std::mt19937 rng{0};

    arrow::AdaptiveIntBuilder builder;
    if (contains_null) {
        auto bitmap = df.combined_bitmap(variable(), evidence());
        auto bitmap_data = bitmap->data();

        int valid_rows = util::bit_util::non_null_count(bitmap, df->num_rows());
        int total_rows = df->num_rows();
        if (valid_rows > m_b) {
            builder.Reserve(m_b);

            std::vector<int64_t> valid_indices;
            valid_indices.reserve(valid_rows);
            for (auto i = 0; i < total_rows; ++i) {
                if (arrow::BitUtil::GetBit(bitmap_data, i)) valid_indices.push_back(i);
            }

            std::shuffle(valid_indices.begin(), valid_indices.end(), rng);

            builder.AppendValues(valid_indices.data(), valid_indices.size());
        } else {
            builder.Reserve(valid_rows);

            for (auto i = 0; i < total_rows; ++i) {
                if (arrow::BitUtil::GetBit(bitmap_data, i)) builder.Append(i);
            }
        }
    } else {
        auto indices = indices_without_replacement(df->num_rows(), m_b, rng);
        RAISE_STATUS_ERROR(builder.Reserve(m_b));
        for (auto i : indices) builder.Append(i);

        // std::cout << "b_indices = " << indices << std::endl;
    }

    Array_ptr take_ind;
    RAISE_STATUS_ERROR(builder.Finish(&take_ind));

    auto u =
        arrow::compute::Take(df.loc(evidence()).record_batch(), take_ind, arrow::compute::TakeOptions::NoBoundsCheck());
    auto v =
        arrow::compute::Take(df.loc(variable()).record_batch(), take_ind, arrow::compute::TakeOptions::NoBoundsCheck());

    auto du = DataFrame(std::move(u).ValueOrDie().record_batch());
    auto dv = DataFrame(std::move(v).ValueOrDie().record_batch());
    return std::make_pair(du, dv);
}

void LSCDE::fit(const DataFrame& df) {
    DataFrame non_null_df = df.loc(variable(), evidence()).filter_null();

    auto type_id = df.same_type(variable(), evidence());
    switch (type_id->id()) {
        case Type::DOUBLE: {
            train_cv<arrow::DoubleType>(non_null_df);
            break;
        }
        case Type::FLOAT: {
            train_cv<arrow::FloatType>(non_null_df);
            break;
        }
        default: {
            throw std::invalid_argument("Wrong data type (" + type_id->ToString() +
                                        ") to fit LSCDE: \"double\" or \"float\" data is expected.");
        }
    }

    m_fitted = true;
}

VectorXd LSCDE::logl(const DataFrame& df) const {
    auto contains_null = df.null_count(variable(), evidence()) > 0;

    auto type_id = df.same_type(variable(), evidence());
    switch (type_id->id()) {
        case Type::DOUBLE:
            if (contains_null) {
                DataFrame non_null_df = df.loc(variable(), evidence()).filter_null();
                auto nonnull_res = _logl<arrow::DoubleType>(non_null_df);

                VectorXd res(df->num_rows());
                auto combined_bitmap = df.combined_bitmap(variable(), evidence());
                auto bitmap_data = combined_bitmap->data();

                for (int i = 0, k = 0; i < df->num_rows(); ++i) {
                    if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                        res(i) = nonnull_res[k++];
                    } else {
                        res(i) = util::nan<double>;
                    }
                }

                return res;
            } else {
                return _logl<arrow::DoubleType>(df);
            }
        case Type::FLOAT:
            if (contains_null) {
                DataFrame non_null_df = df.loc(variable(), evidence()).filter_null();
                auto nonnull_res = _logl<arrow::FloatType>(non_null_df);

                VectorXd res(df->num_rows());
                auto combined_bitmap = df.combined_bitmap(variable(), evidence());
                auto bitmap_data = combined_bitmap->data();

                for (int i = 0, k = 0; i < df->num_rows(); ++i) {
                    if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                        res(i) = nonnull_res[k++];
                    } else {
                        res(i) = util::nan<double>;
                    }
                }

                return res;
            } else {
                return _logl<arrow::FloatType>(df);
            }
        default: {
            throw std::invalid_argument("Wrong data type (" + type_id->ToString() +
                                        ") to fit LSCDE: \"double\" or \"float\" data is expected.");
        }
    }
}

double LSCDE::slogl(const DataFrame& df) const {
    DataFrame non_null_df = df.loc(variable(), evidence()).filter_null();

    auto type_id = df.same_type(variable(), evidence());
    switch (type_id->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(non_null_df).sum();
        case Type::FLOAT:
            return _logl<arrow::FloatType>(non_null_df).sum();
        default: {
            throw std::invalid_argument("Wrong data type (" + type_id->ToString() +
                                        ") to fit LSCDE: \"double\" or \"float\" data is expected.");
        }
    }
}

}  // namespace factors::continuous