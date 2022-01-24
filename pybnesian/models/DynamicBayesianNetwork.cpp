#include <models/DynamicBayesianNetwork.hpp>

namespace models {

template <typename ArrowType>
Array_ptr new_numeric_array(int length) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    arrow::NumericBuilder<ArrowType> builder;
    RAISE_STATUS_ERROR(builder.Resize(length));

    RAISE_STATUS_ERROR(builder.AppendEmptyValues(length));

    std::shared_ptr<ArrayType> out;
    RAISE_STATUS_ERROR(builder.Finish(&out));
    return out;
}

Array_ptr new_discrete_array(const std::vector<std::string>& values, int length) {
    arrow::StringBuilder dict_builder;
    RAISE_STATUS_ERROR(dict_builder.AppendValues(values));

    std::shared_ptr<arrow::StringArray> dictionary;
    RAISE_STATUS_ERROR(dict_builder.Finish(&dictionary));

    arrow::DictionaryBuilder<arrow::StringType> builder(dictionary);
    RAISE_STATUS_ERROR(builder.Resize(length));

    RAISE_STATUS_ERROR(builder.AppendEmptyValues(length));

    std::shared_ptr<arrow::DictionaryArray> out;
    RAISE_STATUS_ERROR(builder.Finish(&out));

    return out;
}

void DynamicBayesianNetwork::add_variable(const std::string& name) {
    if (contains_variable(name)) {
        throw std::invalid_argument("Cannot add variable " + name +
                                    " because a variable with the same name already exists.");
    }

    m_variables.insert(name);

    m_transition->add_node(util::temporal_name(name, 0));

    for (int i = 1; i <= m_markovian_order; ++i) {
        auto new_name = util::temporal_name(name, i);
        m_static->add_node(new_name);
        m_transition->add_interface_node(new_name);
    }
}

void DynamicBayesianNetwork::remove_variable(const std::string& name) {
    if (!contains_variable(name)) {
        throw std::invalid_argument("Cannot remove variable " + name +
                                    " because a variable with the same name do not exist.");
    }

    m_variables.remove(name);

    m_transition->remove_node(util::temporal_name(name, 0));

    for (int i = 1; i <= m_markovian_order; ++i) {
        auto new_name = util::temporal_name(name, i);
        m_static->remove_node(new_name);
        m_transition->remove_interface_node(new_name);
    }
}

VectorXd DynamicBayesianNetwork::logl(const DataFrame& df) const {
    check_fitted();

    if (df->num_rows() < m_markovian_order)
        throw std::invalid_argument(
            "Not enough information. There are less rows in "
            "test DataFrame (" +
            std::to_string(df->num_rows()) +
            ")"
            " than the markovian order of the "
            "DynamicBayesianNetwork (" +
            std::to_string(m_markovian_order) + ")");

    VectorXd ll = VectorXd::Zero(df->num_rows());

    auto static_df = df.slice(0, m_markovian_order);
    auto dstatic_df = create_static_df(static_df, m_markovian_order);

    // Generate logl for the static BN.
    for (int i = 0; i < m_markovian_order; ++i) {
        for (const auto& v : m_variables) {
            const auto& cpd = m_static->cpd(util::temporal_name(v, m_markovian_order - i));
            ll(i) += cpd->slogl(dstatic_df);
        }
    }

    auto temporal_slices = create_temporal_slices(df, m_markovian_order);
    auto dtransition_df = create_transition_df(temporal_slices, m_markovian_order);

    // Generate logl for the transition BN
    for (const auto& v : m_variables) {
        auto name = util::temporal_name(v, 0);

        const auto& cpd = m_transition->cpd(name);
        auto vll = cpd->logl(dtransition_df);

        for (int i = 0; i < vll.rows(); ++i) {
            ll(i + m_markovian_order) += vll(i);
        }
    }

    return ll;
}

double DynamicBayesianNetwork::slogl(const DataFrame& df) const {
    check_fitted();

    if (df->num_rows() < m_markovian_order)
        throw std::invalid_argument(
            "Not enough information. There are less rows in "
            "test DataFrame (" +
            std::to_string(df->num_rows()) +
            ")"
            " than the markovian order of the "
            "DynamicBayesianNetwork (" +
            std::to_string(m_markovian_order) + ")");

    double sll = 0;

    auto static_df = df.slice(0, m_markovian_order);
    auto dstatic_df = create_static_df(static_df, m_markovian_order);

    // Generate slogl for the static BN.
    for (int i = 0; i < m_markovian_order; ++i) {
        for (const auto& v : m_variables) {
            const auto& cpd = m_static->cpd(util::temporal_name(v, m_markovian_order - i));
            sll += cpd->slogl(dstatic_df);
        }
    }

    auto temporal_slices = create_temporal_slices(df, m_markovian_order);
    auto dtransition_df = create_transition_df(temporal_slices, m_markovian_order);

    // Generate logl for the transition BN
    for (const auto& v : m_variables) {
        auto name = util::temporal_name(v, 0);
        const auto& cpd = m_transition->cpd(name);
        sll += cpd->slogl(dtransition_df);
    }

    return sll;
}

std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> DynamicBayesianNetwork::check_same_datatypes() const {
    std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> types;

    for (const auto& v : variables()) {
        std::shared_ptr<arrow::DataType> dt = m_transition->cpd(util::temporal_name(v, 0))->data_type();

        auto dt_id = dt->id();
        for (int i = 1; i <= m_markovian_order; ++i) {
            auto name = util::temporal_name(v, i);

            if (dt_id != m_static->cpd(name)->data_type()->id())
                throw std::invalid_argument(
                    "Data type for transition Bayesian network node " + util::temporal_name(v, 0) + " [" +
                    dt->ToString() + "] is different from data type of static Bayesian network node " +
                    util::temporal_name(v, i) + "[" + m_static->cpd(name)->data_type()->ToString() + "]");

            if (dt_id != m_transition->cpd(name)->data_type()->id())
                throw std::invalid_argument(
                    "Data type for transition Bayesian network node " + util::temporal_name(v, 0) + " [" +
                    dt->ToString() + "] is different from data type of transition Bayesian network node " +
                    util::temporal_name(v, i) + "[" + m_transition->cpd(name)->data_type()->ToString() + "]");
        }

        types.insert({v, dt});
    }

    return types;
}

std::vector<std::string> discretefactor_possible_values(const DynamicBayesianNetwork& dbn,
                                                        const std::string& variable) {
    const auto& cpd = dbn.transition_bn().cpd(util::temporal_name(variable, 0));
    const auto& discrete_cpd = std::static_pointer_cast<DiscreteFactor>(cpd);

    std::vector<std::string> values = discrete_cpd->variable_values();

    for (int i = 1; i < dbn.markovian_order(); ++i) {
        const auto& cpd = dbn.static_bn().cpd(util::temporal_name(variable, i));
        const auto& discrete_cpd = std::static_pointer_cast<DiscreteFactor>(cpd);

        if (values != discrete_cpd->variable_values()) {
            throw std::invalid_argument("CPD of transition Bayesian network node " + util::temporal_name(variable, 0) +
                                        " have different categories than "
                                        "static Bayesian network node " +
                                        util::temporal_name(variable, i) + ".");
        }
    }

    return values;
}

DataFrame generate_empty_dataframe(const DynamicBayesianNetwork& dbn,
                                   int n,
                                   std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> types) {
    std::vector<Array_ptr> columns;
    std::vector<Field_ptr> fields;

    for (const auto& variable : dbn.variables()) {
        auto type = types.at(variable);
        switch (type->id()) {
            case Type::DOUBLE: {
                columns.push_back(new_numeric_array<arrow::DoubleType>(n));
                break;
            }
            case Type::FLOAT: {
                columns.push_back(new_numeric_array<arrow::FloatType>(n));
                break;
            }
            case Type::DICTIONARY: {
                auto variable_values = discretefactor_possible_values(dbn, variable);
                columns.push_back(new_discrete_array(variable_values, n));
                break;
            }
            default:
                throw std::invalid_argument("Data type not supported for sampling.");
        }

        fields.push_back(arrow::field(variable, type));
    }

    auto schema = arrow::schema(fields);

    return DataFrame(arrow::RecordBatch::Make(schema, n, columns));
}

template <typename ArrowType>
void sample_discrete_static_bn(const std::string& variable,
                               int max_length,
                               const DataFrame& static_sample,
                               const DynamicBayesianNetwork& dbn,
                               Array_ptr& output_indices) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto raw_output = output_indices->data()->template GetMutableValues<typename ArrowType::c_type>(1);
    for (int i = 0; i < max_length; ++i) {
        auto name = util::temporal_name(variable, dbn.markovian_order() - i);

        auto input = static_sample.col(name);
        auto dict_input = std::static_pointer_cast<arrow::DictionaryArray>(input);
        auto input_indices = dict_input->indices();

        auto dwn_input_indices = std::static_pointer_cast<ArrayType>(input_indices);
        raw_output[i] = dwn_input_indices->Value(0);
    }
}

void sample_static_bn(DataFrame& df,
                      const std::unordered_map<std::string, std::shared_ptr<arrow::DataType>>& types,
                      const DynamicBayesianNetwork& dbn,
                      int n,
                      unsigned int seed) {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    auto static_sample = dbn.static_bn().sample(1, seed);

    auto max_length = std::min(dbn.markovian_order(), n);

    for (const auto& v : dbn.variables()) {
        switch (types.at(v)->id()) {
            case Type::DOUBLE: {
                auto raw_col = df.template mutable_data<arrow::DoubleType>(v);

                for (int i = 0; i < max_length; ++i) {
                    auto name = util::temporal_name(v, dbn.markovian_order() - i);
                    raw_col[i] = *static_sample.template data<arrow::DoubleType>(name);
                }
                break;
            }
            case Type::FLOAT: {
                auto raw_col = df.mutable_data<arrow::FloatType>(v);

                for (int i = 0; i < max_length; ++i) {
                    auto name = util::temporal_name(v, dbn.markovian_order() - i);
                    raw_col[i] = *static_sample.template data<arrow::FloatType>(name);
                }
                break;
            }
            case Type::DICTIONARY: {
                auto col = df.col(v);
                auto dict_col = std::static_pointer_cast<arrow::DictionaryArray>(col);
                auto indices = dict_col->indices();
                switch (indices->type_id()) {
                    case Type::INT8: {
                        sample_discrete_static_bn<arrow::Int8Type>(v, max_length, static_sample, dbn, indices);
                        break;
                    }
                    case Type::INT16: {
                        sample_discrete_static_bn<arrow::Int16Type>(v, max_length, static_sample, dbn, indices);
                        break;
                    }
                    case Type::INT32: {
                        sample_discrete_static_bn<arrow::Int32Type>(v, max_length, static_sample, dbn, indices);
                        break;
                    }
                    case Type::INT64: {
                        sample_discrete_static_bn<arrow::Int64Type>(v, max_length, static_sample, dbn, indices);
                        break;
                    }
                    default:
                        throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
                }
                break;
            }
            default:
                throw std::runtime_error("Wrong data type for variable " + v + ": " + types.at(v)->ToString());
        }
    }
}

void sample_transition_bn(DataFrame& df,
                          const std::unordered_map<std::string, std::shared_ptr<arrow::DataType>>& types,
                          const DynamicBayesianNetwork& dbn,
                          int n,
                          unsigned int seed) {
    auto top_sort = dbn.transition_bn().graph().topological_sort();

    for (int i = dbn.markovian_order(); i < n; ++i) {
        auto evidence_slice = df.slice(i - dbn.markovian_order(), dbn.markovian_order() + 1);
        DynamicDataFrame slice_ddf(evidence_slice, dbn.markovian_order());
        for (const auto& full_variable : top_sort) {
            // Remove the suffix "_t_0"
            auto variable = full_variable.substr(0, full_variable.size() - 4);
            const auto cpd = dbn.transition_bn().cpd(full_variable);

            auto sampled = cpd->sample(1, slice_ddf.transition_df(), seed + i);

            switch (types.at(variable)->id()) {
                case Type::DOUBLE: {
                    auto raw_col = df.template mutable_data<arrow::DoubleType>(variable);

                    switch (sampled->type_id()) {
                        case Type::DOUBLE: {
                            auto dwn_sampled = std::static_pointer_cast<arrow::DoubleArray>(sampled);
                            raw_col[i] = dwn_sampled->Value(0);
                            break;
                        }
                        case Type::FLOAT: {
                            auto dwn_sampled = std::static_pointer_cast<arrow::FloatArray>(sampled);
                            raw_col[i] = dwn_sampled->Value(0);
                            break;
                        }
                        default:
                            throw std::runtime_error("Wrong sampled data type for variable " + variable);
                    }

                    break;
                }
                case Type::FLOAT: {
                    auto raw_col = df.template mutable_data<arrow::FloatType>(variable);

                    switch (sampled->type_id()) {
                        case Type::DOUBLE: {
                            auto dwn_sampled = std::static_pointer_cast<arrow::DoubleArray>(sampled);
                            raw_col[i] = dwn_sampled->Value(0);
                            break;
                        }
                        case Type::FLOAT: {
                            auto dwn_sampled = std::static_pointer_cast<arrow::FloatArray>(sampled);
                            raw_col[i] = dwn_sampled->Value(0);
                            break;
                        }
                        default:
                            throw std::runtime_error("Wrong sampled data type for variable " + variable);
                    }

                    break;
                }
                case Type::DICTIONARY: {
                    auto col = df.col(variable);
                    auto dict_col = std::static_pointer_cast<arrow::DictionaryArray>(col);
                    auto indices = dict_col->indices();

                    switch (indices->type_id()) {
                        case Type::INT8: {
                            using CType = typename arrow::Int8Type::c_type;
                            auto raw_output = indices->data()->template GetMutableValues<CType>(1);

                            auto dict_sampled = std::static_pointer_cast<arrow::DictionaryArray>(sampled);
                            auto indices_sampled = dict_sampled->indices();
                            auto dwn_indices_sampled = std::static_pointer_cast<arrow::Int8Array>(indices_sampled);

                            raw_output[i] = dwn_indices_sampled->Value(0);
                            break;
                        }
                        case Type::INT16: {
                            using CType = typename arrow::Int16Type::c_type;
                            auto raw_output = indices->data()->template GetMutableValues<CType>(1);

                            auto dict_sampled = std::static_pointer_cast<arrow::DictionaryArray>(sampled);
                            auto indices_sampled = dict_sampled->indices();
                            auto dwn_indices_sampled = std::static_pointer_cast<arrow::Int16Array>(indices_sampled);

                            raw_output[i] = dwn_indices_sampled->Value(0);

                            break;
                        }
                        case Type::INT32: {
                            using CType = typename arrow::Int32Type::c_type;
                            auto raw_output = indices->data()->template GetMutableValues<CType>(1);

                            auto dict_sampled = std::static_pointer_cast<arrow::DictionaryArray>(sampled);
                            auto indices_sampled = dict_sampled->indices();
                            auto dwn_indices_sampled = std::static_pointer_cast<arrow::Int32Array>(indices_sampled);

                            raw_output[i] = dwn_indices_sampled->Value(0);

                            break;
                        }
                        case Type::INT64: {
                            using CType = typename arrow::Int64Type::c_type;
                            auto raw_output = indices->data()->template GetMutableValues<CType>(1);

                            auto dict_sampled = std::static_pointer_cast<arrow::DictionaryArray>(sampled);
                            auto indices_sampled = dict_sampled->indices();
                            auto dwn_indices_sampled = std::static_pointer_cast<arrow::Int64Array>(indices_sampled);

                            raw_output[i] = dwn_indices_sampled->Value(0);

                            break;
                        }
                        default:
                            throw std::invalid_argument(
                                "Wrong indices array type of DictionaryArray "
                                "for variable " +
                                variable);
                    }

                    break;
                }
                default:
                    throw std::runtime_error("Wrong data type for variable " + variable + ": " +
                                             types.at(variable)->ToString());
            }
        }
    }
}

DataFrame DynamicBayesianNetwork::sample(int n, unsigned int seed) const {
    check_fitted();
    auto types = check_same_datatypes();

    auto sampled = generate_empty_dataframe(*this, n, types);

    sample_static_bn(sampled, types, *this, n, seed);
    sample_transition_bn(sampled, types, *this, n, seed);

    return sampled;
}

py::tuple DynamicBayesianNetwork::__getstate__() const {
    m_static->set_include_cpd(m_include_cpd);
    m_transition->set_include_cpd(m_include_cpd);
    return py::make_tuple(m_variables.elements(), m_markovian_order, m_static, m_transition);
}

void __nonderived_dbn_setstate__(py::object& self, py::tuple& t) {
    if (t.size() != 4) throw std::runtime_error("Not valid DynamicBayesianNetwork");

    auto variables = t[0].cast<std::vector<std::string>>();
    auto markovian_order = t[1].cast<int>();
    // The BNs could be Python BNs, so ensure the Python objects are kept alive.
    auto static_bn = t[2].cast<std::shared_ptr<BayesianNetworkBase>>();
    BayesianNetworkBase::keep_python_alive(static_bn);
    auto transition_bn = t[3].cast<std::shared_ptr<ConditionalBayesianNetworkBase>>();
    ConditionalBayesianNetworkBase::keep_python_alive(transition_bn);

    auto pydbntype = py::type::of<DynamicBayesianNetwork>();
    pydbntype.attr("__init__")(self, variables, markovian_order, static_bn, transition_bn);
}

void DynamicBayesianNetwork::save(std::string name, bool include_cpd) const {
    m_include_cpd = include_cpd;
    auto open = py::module::import("io").attr("open");

    if (name.size() < 7 || name.substr(name.size() - 7) != ".pickle") name += ".pickle";

    auto file = open(name, "wb");
    py::module_::import("pickle").attr("dump")(py::cast(this), file, 2);
    file.attr("close")();
}

}  // namespace models