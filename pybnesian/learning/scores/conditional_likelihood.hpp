#ifndef PYBNESIAN_LEARNING_SCORES_CONDITIONAL_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_CONDITIONAL_LIKELIHOOD_HPP

#include <learning/scores/scores.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/scores/cv_likelihood.hpp>

using learning::scores::HoldoutLikelihood, learning::scores::CVLikelihood;

namespace learning::scores {

class ConditionalLikelihood : public ValidatedScore {
public:
    ConditionalLikelihood(const DataFrame& df,
                          const std::string& class_name,
                          double test_ratio = 0.2,
                          int k = 10,
                          unsigned int seed = std::random_device{}())
        : m_class(class_name), m_holdout(df, test_ratio, seed), m_cv(m_holdout.training_data(), k, seed) {
        if (!df.is_discrete(class_name)) {
            throw std::invalid_argument("Class column " + class_name + " must be categorical.");
        }
    }

    using ValidatedScore::local_score;

    bool is_decomposable() const override { return false; }

    template <typename IndicesArrowType>
    double score_cv_cll_penalty(const BayesianNetworkBase& model) const {
        using CType = typename IndicesArrowType::c_type;

        std::unordered_set<std::string> needed_cols;
        needed_cols.insert(m_class);
        auto class_parents = model.parents(m_class);
        needed_cols.insert(class_parents.begin(), class_parents.end());
        auto class_children = model.children(m_class);
        needed_cols.insert(class_children.begin(), class_children.end());

        for (const auto& ch : class_children) {
            auto pch = model.parents(ch);
            needed_cols.insert(pch.begin(), pch.end());
        }

        double s = 0;

        std::vector<std::string> fixed_cpds(class_children.begin(), class_children.end());
        fixed_cpds.push_back(m_class);

        for (auto [train_df, test_df] : m_cv.loc(needed_cols)) {
            auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(test_df.col(m_class));
            auto original_class_field = test_df->schema()->GetFieldByName(m_class);
            auto class_index = test_df.index(m_class);

            auto indices_array = dataset::copy_array(dwn_original_class->indices());
            auto raw_indices = indices_array->data()->template GetMutableValues<CType>(1);

            auto num_classes = dwn_original_class->dictionary()->length();

            MatrixXd logl_class = MatrixXd::Zero(test_df->num_rows(), num_classes);

            for (auto k = 0; k < num_classes; ++k) {
                std::fill(raw_indices, raw_indices + indices_array->length(), k);

                auto new_class_column = std::make_shared<arrow::DictionaryArray>(
                    dwn_original_class->type(), indices_array, dwn_original_class->dictionary());

                auto res_new_df = test_df->SetColumn(class_index, original_class_field, new_class_column);
                auto new_test_df = std::move(res_new_df).ValueOrDie();

                for (const auto& fv : fixed_cpds) {
                    auto fixed_cpd = model.node_type(fv)->new_factor(model, fv, model.parents(fv));
                    fixed_cpd->fit(train_df);
                    logl_class.col(k) += fixed_cpd->logl(new_test_df);
                }
            }

            if (test_df.null_count(fixed_cpds) == 0) {
                auto max_values = logl_class.rowwise().maxCoeff();
                s += max_values.sum() + (logl_class.colwise() - max_values).array().exp().rowwise().sum().log().sum();
            } else {
                auto combined_bitmap = test_df.combined_bitmap(fixed_cpds);
                auto bitmap_data = combined_bitmap->data();

                for (int i = 0; i < test_df->num_rows(); ++i) {
                    if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                        auto row_max = logl_class.row(i).maxCoeff();
                        s += row_max + std::log((logl_class.row(i).array() - row_max).exp().sum());
                    }
                }
            }
        }

        return s;
    }

    template <typename IndicesArrowType>
    double score_holdout_cll_penalty(const BayesianNetworkBase& model) const {
        using CType = typename IndicesArrowType::c_type;

        std::unordered_set<std::string> needed_cols;
        needed_cols.insert(m_class);
        auto class_parents = model.parents(m_class);
        needed_cols.insert(class_parents.begin(), class_parents.end());
        auto class_children = model.children(m_class);
        needed_cols.insert(class_children.begin(), class_children.end());

        for (const auto& ch : class_children) {
            auto pch = model.parents(ch);
            needed_cols.insert(pch.begin(), pch.end());
        }

        std::vector<std::string> fixed_cpds(class_children.begin(), class_children.end());
        fixed_cpds.push_back(m_class);

        const auto& train_df = m_holdout.training_data();
        const auto& test_df = m_holdout.test_data();

        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(test_df.col(m_class));
        auto original_class_field = test_df->schema()->GetFieldByName(m_class);
        auto class_index = test_df.index(m_class);

        auto indices_array = dataset::copy_array(dwn_original_class->indices());
        auto raw_indices = indices_array->data()->template GetMutableValues<CType>(1);

        auto num_classes = dwn_original_class->dictionary()->length();

        MatrixXd logl_class = MatrixXd::Zero(test_df->num_rows(), num_classes);

        for (auto k = 0; k < num_classes; ++k) {
            std::fill(raw_indices, raw_indices + indices_array->length(), k);

            auto new_class_column = std::make_shared<arrow::DictionaryArray>(
                dwn_original_class->type(), indices_array, dwn_original_class->dictionary());

            auto res_new_df = test_df->SetColumn(class_index, original_class_field, new_class_column);
            auto new_test_df = std::move(res_new_df).ValueOrDie();

            for (const auto& fv : fixed_cpds) {
                auto fixed_cpd = model.node_type(fv)->new_factor(model, fv, model.parents(fv));
                fixed_cpd->fit(train_df);
                logl_class.col(k) += fixed_cpd->logl(new_test_df);
            }
        }

        if (test_df.null_count(fixed_cpds) == 0) {
            auto max_values = logl_class.rowwise().maxCoeff();
            return max_values.sum() + (logl_class.colwise() - max_values).array().exp().rowwise().sum().log().sum();
        } else {
            auto combined_bitmap = test_df.combined_bitmap(fixed_cpds);
            auto bitmap_data = combined_bitmap->data();

            double s = 0;
            for (int i = 0; i < test_df->num_rows(); ++i) {
                if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                    auto row_max = logl_class.row(i).maxCoeff();
                    s += row_max + std::log((logl_class.row(i).array() - row_max).exp().sum());
                }
            }

            return s;
        }
    }

    double score(const BayesianNetworkBase& model) const override {
        double s = 0;

        auto class_children = model.children(m_class);
        for (const auto& node : model.nodes()) {
            bool is_class_children =
                std::find(class_children.begin(), class_children.end(), node) != class_children.end();
            if (node == m_class || is_class_children) {
                auto cpd = model.underlying_node_type(m_cv.data(), node)->new_factor(model, node, model.parents(node));

                for (auto [train_df, test_df] : m_cv.loc(node, model.parents(node))) {
                    cpd->fit(train_df);
                    s += cpd->slogl(test_df);
                }
            }
        }

        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                s -= score_cv_cll_penalty<arrow::Int8Type>(model);
                break;
            case Type::INT16:
                s -= score_cv_cll_penalty<arrow::Int16Type>(model);
                break;
            case Type::INT32:
                s -= score_cv_cll_penalty<arrow::Int32Type>(model);
                break;
            case Type::INT64:
                s -= score_cv_cll_penalty<arrow::Int64Type>(model);
                break;
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }

        return s;
    }

    double vscore(const BayesianNetworkBase& model) const override {
        double s = 0;

        auto class_children = model.children(m_class);
        for (const auto& node : model.nodes()) {
            bool is_class_children =
                std::find(class_children.begin(), class_children.end(), node) != class_children.end();
            if (node == m_class || is_class_children) {
                auto cpd = model.underlying_node_type(m_cv.data(), node)->new_factor(model, node, model.parents(node));

                cpd->fit(m_holdout.training_data());
                s += cpd->slogl(m_holdout.test_data());
            }
        }

        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                s -= score_holdout_cll_penalty<arrow::Int8Type>(model);
                break;
            case Type::INT16:
                s -= score_holdout_cll_penalty<arrow::Int16Type>(model);
                break;
            case Type::INT32:
                s -= score_holdout_cll_penalty<arrow::Int32Type>(model);
                break;
            case Type::INT64:
                s -= score_holdout_cll_penalty<arrow::Int64Type>(model);
                break;
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }

        return s;
    }


    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    double local_score(const BayesianNetworkBase& model,
                       const FactorType& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    std::string ToString() const override { return "ConditionalLikelihood"; }

    bool has_variables(const std::string& name) const override { return m_cv.data().has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_cv.data().has_columns(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override { return has_variables(model.nodes()); }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        return has_variables(model.joint_nodes());
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override;

    double vlocal_score(const BayesianNetworkBase& model,
                        const FactorType& variable_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override;

    template<typename IndicesArrowType>
    std::pair<Array_ptr, MatrixXd> predict_template(const DataFrame& df, const BayesianNetworkBase& model) const;
    std::pair<Array_ptr, MatrixXd> predict(const DataFrame& df, const BayesianNetworkBase& model) const;

    const DataFrame& training_data() { return m_holdout.training_data(); }
    const DataFrame& validation_data() { return m_holdout.test_data(); }
    const HoldOut& holdout() const { return m_holdout; }
    const CrossValidation& cv() const { return m_cv; }

    DataFrame data() const override { return m_cv.data(); }

private:
    template <typename IndicesArrowType>
    double cll_penalty(const BayesianNetworkBase& model,
                       const FactorType& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents,
                       const std::vector<std::string>& fixed_cpds) const;

    double cll_penalty(const BayesianNetworkBase& model,
                       const FactorType& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents,
                       const std::vector<std::string>& fixed_cpds) const;

    std::string m_class;
    HoldOut m_holdout;
    CrossValidation m_cv;
};

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_CONDITIONAL_LIKELIHOOD_HPP