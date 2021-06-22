#include <learning/scores/conditional_likelihood.hpp>

namespace learning::scores {

template <typename IndicesArrowType>
double vcll_penalty(const DataFrame& train_df,
                    const DataFrame& test_df,
                    const BayesianNetworkBase& model,
                    const FactorType& variable_type,
                    const std::string& variable,
                    const std::vector<std::string>& parents,
                    const std::string& class_variable,
                    const std::vector<std::string>& fixed_cpds) {
    using CType = typename IndicesArrowType::c_type;

    auto cpd = variable_type.new_factor(model, variable, parents);
    cpd->fit(train_df);
    double loglik = cpd->slogl(test_df);

    auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(test_df.col(class_variable));
    auto original_class_field = test_df->schema()->GetFieldByName(class_variable);
    auto class_index = test_df.index(class_variable);

    auto indices_array = dataset::copy_array(dwn_original_class->indices());
    auto raw_indices = indices_array->data()->template GetMutableValues<CType>(1);

    auto num_classes = dwn_original_class->dictionary()->length();

    MatrixXd logl_class(test_df->num_rows(), num_classes);

    for (auto k = 0; k < num_classes; ++k) {
        std::fill(raw_indices, raw_indices + indices_array->length(), k);

        auto new_class_column = std::make_shared<arrow::DictionaryArray>(
            dwn_original_class->type(), indices_array, dwn_original_class->dictionary());

        auto res_new_df = test_df->SetColumn(class_index, original_class_field, new_class_column);
        auto new_test_df = std::move(res_new_df).ValueOrDie();

        logl_class.col(k) = cpd->logl(new_test_df);

        for (const auto& fv : fixed_cpds) {
            auto fixed_cpd = model.node_type(fv)->new_factor(model, fv, model.parents(fv));
            fixed_cpd->fit(train_df);
            logl_class.col(k) += fixed_cpd->logl(new_test_df);
        }
    }

    if (test_df.null_count(variable, fixed_cpds) == 0) {
        auto max_values = logl_class.rowwise().maxCoeff();
        loglik -= max_values.sum() + (logl_class.colwise() - max_values).array().exp().rowwise().sum().log().sum();
    } else {
        auto combined_bitmap = test_df.combined_bitmap(variable, fixed_cpds);
        auto bitmap_data = combined_bitmap->data();

        for (int i = 0; i < test_df->num_rows(); ++i) {
            if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                auto row_max = logl_class.row(i).maxCoeff();
                loglik -= row_max + std::log((logl_class.row(i).array() - row_max).exp().sum());
            }
        }
    }

    return loglik;
}

double vcll_penalty(const DataFrame& train_df,
                    const DataFrame& test_df,
                    const BayesianNetworkBase& model,
                    const FactorType& variable_type,
                    const std::string& variable,
                    const std::vector<std::string>& parents,
                    const std::string& class_variable,
                    const std::vector<std::string>& fixed_cpds) {
    auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(test_df.col(class_variable));

    switch (dwn_original_class->indices()->type_id()) {
        case Type::INT8:
            return vcll_penalty<arrow::Int8Type>(
                train_df, test_df, model, variable_type, variable, parents, class_variable, fixed_cpds);
        case Type::INT16:
            return vcll_penalty<arrow::Int16Type>(
                train_df, test_df, model, variable_type, variable, parents, class_variable, fixed_cpds);
        case Type::INT32:
            return vcll_penalty<arrow::Int32Type>(
                train_df, test_df, model, variable_type, variable, parents, class_variable, fixed_cpds);
        case Type::INT64:
            return vcll_penalty<arrow::Int64Type>(
                train_df, test_df, model, variable_type, variable, parents, class_variable, fixed_cpds);
        default:
            throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
    }
}

template <typename IndicesArrowType>
double ConditionalLikelihood::cll_penalty(const BayesianNetworkBase& model,
                                          const FactorType& variable_type,
                                          const std::string& variable,
                                          const std::vector<std::string>& parents,
                                          const std::vector<std::string>& fixed_cpds) const {
    double loglik = 0;

    std::unordered_set<std::string> needed_columns;
    needed_columns.insert(m_class);
    needed_columns.insert(parents.begin(), parents.end());

    auto children_class = model.children(m_class);
    needed_columns.insert(children_class.begin(), children_class.end());

    // Include the markov blanket
    if (variable == m_class) {
        for (const auto& ch : children_class) {
            auto pch = model.parents(ch);
            needed_columns.insert(pch.begin(), pch.end());
        }
    } else {
        needed_columns.insert(variable);

        auto parents_class = model.parents(m_class);
        needed_columns.insert(parents_class.begin(), parents_class.end());

        for (const auto& ch : children_class) {
            if (ch != variable) {
                auto pch = model.parents(ch);
                needed_columns.insert(pch.begin(), pch.end());
            }
        }
    }

    for (auto [train_df, test_df] : m_cv.loc(needed_columns)) {
        loglik += vcll_penalty<IndicesArrowType>(
            train_df, test_df, model, variable_type, variable, parents, m_class, fixed_cpds);
    }

    return loglik;
}

double ConditionalLikelihood::cll_penalty(const BayesianNetworkBase& model,
                                          const FactorType& variable_type,
                                          const std::string& variable,
                                          const std::vector<std::string>& parents,
                                          const std::vector<std::string>& fixed_cpds) const {
    auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

    switch (dwn_original_class->indices()->type_id()) {
        case Type::INT8:
            return cll_penalty<arrow::Int8Type>(model, variable_type, variable, parents, fixed_cpds);
        case Type::INT16:
            return cll_penalty<arrow::Int16Type>(model, variable_type, variable, parents, fixed_cpds);
        case Type::INT32:
            return cll_penalty<arrow::Int32Type>(model, variable_type, variable, parents, fixed_cpds);
        case Type::INT64:
            return cll_penalty<arrow::Int64Type>(model, variable_type, variable, parents, fixed_cpds);
        default:
            throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
    }
}

double ConditionalLikelihood::local_score(const BayesianNetworkBase& model,
                                          const std::string& variable,
                                          const std::vector<std::string>& parents) const {
    if (variable == m_class) {
        return cll_penalty(
            model, *model.underlying_node_type(m_cv.data(), variable), variable, parents, model.children(variable));
    } else if (std::find(parents.begin(), parents.end(), m_class) != parents.end()) {
        auto fixed_cpds = model.children(m_class);

        auto it = std::find(fixed_cpds.begin(), fixed_cpds.end(), variable);
        if (it != fixed_cpds.end()) util::iter_swap_remove(fixed_cpds, it);

        fixed_cpds.push_back(m_class);

        return cll_penalty(model, *model.underlying_node_type(m_cv.data(), variable), variable, parents, fixed_cpds);
    } else {
        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        auto variable_parents = model.parents(variable);
        auto actual_model = model.clone();

        if (std::find(variable_parents.begin(), variable_parents.end(), m_class) != variable_parents.end()) {
            actual_model->remove_arc(m_class, variable);
        }

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                return -score_cv_cll_penalty<arrow::Int8Type>(*actual_model);
            case Type::INT16:
                return -score_cv_cll_penalty<arrow::Int16Type>(*actual_model);
            case Type::INT32:
                return -score_cv_cll_penalty<arrow::Int32Type>(*actual_model);
            case Type::INT64:
                return -score_cv_cll_penalty<arrow::Int64Type>(*actual_model);
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }
}

double ConditionalLikelihood::local_score(const BayesianNetworkBase& model,
                                          const FactorType& variable_type,
                                          const std::string& variable,
                                          const std::vector<std::string>& parents) const {
    if (variable == m_class) {
        return cll_penalty(model, variable_type, variable, parents, model.children(variable));
    } else if (std::find(parents.begin(), parents.end(), m_class) != parents.end()) {
        auto fixed_cpds = model.children(m_class);

        auto it = std::find(fixed_cpds.begin(), fixed_cpds.end(), variable);
        if (it != fixed_cpds.end()) util::iter_swap_remove(fixed_cpds, it);

        fixed_cpds.push_back(m_class);
        return cll_penalty(model, variable_type, variable, parents, fixed_cpds);
    } else {
        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        auto variable_parents = model.parents(variable);
        auto actual_model = model.clone();

        if (std::find(variable_parents.begin(), variable_parents.end(), m_class) != variable_parents.end()) {
            actual_model->remove_arc(m_class, variable);
        }

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                return -score_cv_cll_penalty<arrow::Int8Type>(*actual_model);
            case Type::INT16:
                return -score_cv_cll_penalty<arrow::Int16Type>(*actual_model);
            case Type::INT32:
                return -score_cv_cll_penalty<arrow::Int32Type>(*actual_model);
            case Type::INT64:
                return -score_cv_cll_penalty<arrow::Int64Type>(*actual_model);
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }
}

double ConditionalLikelihood::vlocal_score(const BayesianNetworkBase& model,
                                           const std::string& variable,
                                           const std::vector<std::string>& parents) const {
    if (variable == m_class) {
        return vcll_penalty(m_holdout.training_data(),
                            m_holdout.test_data(),
                            model,
                            *model.underlying_node_type(m_cv.data(), variable),
                            variable,
                            parents,
                            m_class,
                            model.children(variable));
    } else if (std::find(parents.begin(), parents.end(), m_class) != parents.end()) {
        auto fixed_cpds = model.children(m_class);

        auto it = std::find(fixed_cpds.begin(), fixed_cpds.end(), variable);
        if (it != fixed_cpds.end()) util::iter_swap_remove(fixed_cpds, it);

        fixed_cpds.push_back(m_class);
        return vcll_penalty(m_holdout.training_data(),
                            m_holdout.test_data(),
                            model,
                            *model.underlying_node_type(m_cv.data(), variable),
                            variable,
                            parents,
                            m_class,
                            fixed_cpds);
    } else {
        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        auto variable_parents = model.parents(variable);
        auto actual_model = model.clone();

        if (std::find(variable_parents.begin(), variable_parents.end(), m_class) != variable_parents.end()) {
            actual_model->remove_arc(m_class, variable);
        }

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                return -score_holdout_cll_penalty<arrow::Int8Type>(*actual_model);
            case Type::INT16:
                return -score_holdout_cll_penalty<arrow::Int16Type>(*actual_model);
            case Type::INT32:
                return -score_holdout_cll_penalty<arrow::Int32Type>(*actual_model);
            case Type::INT64:
                return -score_holdout_cll_penalty<arrow::Int64Type>(*actual_model);
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }
}

double ConditionalLikelihood::vlocal_score(const BayesianNetworkBase& model,
                                           const FactorType& variable_type,
                                           const std::string& variable,
                                           const std::vector<std::string>& parents) const {
    if (variable == m_class) {
        return vcll_penalty(m_holdout.training_data(),
                            m_holdout.test_data(),
                            model,
                            variable_type,
                            variable,
                            parents,
                            m_class,
                            model.children(variable));
    } else if (std::find(parents.begin(), parents.end(), m_class) != parents.end()) {
        auto fixed_cpds = model.children(m_class);

        auto it = std::find(fixed_cpds.begin(), fixed_cpds.end(), variable);
        if (it != fixed_cpds.end()) util::iter_swap_remove(fixed_cpds, it);

        fixed_cpds.push_back(m_class);
        return vcll_penalty(m_holdout.training_data(),
                            m_holdout.test_data(),
                            model,
                            variable_type,
                            variable,
                            parents,
                            m_class,
                            fixed_cpds);
    } else {
        auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

        auto variable_parents = model.parents(variable);
        auto actual_model = model.clone();

        if (std::find(variable_parents.begin(), variable_parents.end(), m_class) != variable_parents.end()) {
            actual_model->remove_arc(m_class, variable);
        }

        switch (dwn_original_class->indices()->type_id()) {
            case Type::INT8:
                return -score_holdout_cll_penalty<arrow::Int8Type>(*actual_model);
            case Type::INT16:
                return -score_holdout_cll_penalty<arrow::Int16Type>(*actual_model);
            case Type::INT32:
                return -score_holdout_cll_penalty<arrow::Int32Type>(*actual_model);
            case Type::INT64:
                return -score_holdout_cll_penalty<arrow::Int64Type>(*actual_model);
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }
}

template <typename IndicesArrowType>
std::pair<Array_ptr, MatrixXd> ConditionalLikelihood::predict_template(const DataFrame& df,
                                                                       const BayesianNetworkBase& model) const {
    using CType = typename IndicesArrowType::c_type;

    auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));
    auto num_classes = dwn_original_class->dictionary()->length();

    auto test_class = std::static_pointer_cast<arrow::DictionaryArray>(df.col(m_class));
    auto test_class_field = df->schema()->GetFieldByName(m_class);
    auto class_index = df.index(m_class);

    auto indices_array = dataset::copy_array(test_class->indices());
    auto raw_indices = indices_array->data()->template GetMutableValues<CType>(1);

    MatrixXd logl_class = MatrixXd::Zero(df->num_rows(), num_classes);

    auto class_children = model.children(m_class);
    std::vector<std::string> fixed_cpds(class_children.begin(), class_children.end());
    fixed_cpds.push_back(m_class);

    for (auto k = 0; k < num_classes; ++k) {
        std::fill(raw_indices, raw_indices + indices_array->length(), k);

        auto new_class_column = std::make_shared<arrow::DictionaryArray>(
            dwn_original_class->type(), indices_array, dwn_original_class->dictionary());

        auto res_new_df = df->SetColumn(class_index, test_class_field, new_class_column);
        auto new_test_df = std::move(res_new_df).ValueOrDie();

        for (const auto& fv : fixed_cpds) {
            auto fixed_cpd = model.node_type(fv)->new_factor(model, fv, model.parents(fv));
            fixed_cpd->fit(m_cv.data());
            logl_class.col(k) += fixed_cpd->logl(new_test_df);
        }
    }

    for (int i = 0; i < indices_array->length(); ++i) {
        int max_index;
        logl_class.row(i).maxCoeff(&max_index);
        raw_indices[i] = max_index;
    }

    return std::make_pair(std::make_shared<arrow::DictionaryArray>(
                              dwn_original_class->type(), indices_array, dwn_original_class->dictionary()),
                          logl_class);
}

std::pair<Array_ptr, MatrixXd> ConditionalLikelihood::predict(const DataFrame& df,
                                                              const BayesianNetworkBase& model) const {
    auto dwn_original_class = std::static_pointer_cast<arrow::DictionaryArray>(m_cv.data().col(m_class));

    switch (dwn_original_class->indices()->type_id()) {
        case Type::INT8:
            return predict_template<arrow::Int8Type>(df, model);
        case Type::INT16:
            return predict_template<arrow::Int16Type>(df, model);
        case Type::INT32:
            return predict_template<arrow::Int32Type>(df, model);
        case Type::INT64:
            return predict_template<arrow::Int64Type>(df, model);
        default:
            throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
    }
}

}  // namespace learning::scores