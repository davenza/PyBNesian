#ifndef PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP

#include <dataset/dynamic_dataset.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <util/temporal.hpp>

using dataset::DynamicDataFrame;
using factors::continuous::LinearGaussianCPD, factors::continuous::CKDE,
      factors::continuous::SemiparametricCPD, factors::discrete::DiscreteFactor;
using models::BayesianNetworkBase;

namespace models {

    class DynamicBayesianNetworkBase : public clone_inherit<abstract_class<DynamicBayesianNetworkBase>> {
    public:
        virtual ~DynamicBayesianNetworkBase() = default;
        virtual BayesianNetworkBase& static_bn() = 0;
        virtual const BayesianNetworkBase& static_bn() const = 0;
        virtual ConditionalBayesianNetworkBase& transition_bn() = 0;
        virtual const ConditionalBayesianNetworkBase& transition_bn() const = 0;

        virtual int markovian_order() const = 0;
        virtual int num_variables() const = 0;
        virtual const std::vector<std::string>& variables() const = 0;
        virtual bool contains_variable(const std::string& name) const = 0;
        virtual void add_variable(const std::string& name) = 0;
        virtual void remove_variable(const std::string& name) = 0;
        virtual bool fitted() const = 0;
        virtual void fit(const DataFrame& df) = 0;
        virtual VectorXd logl(const DataFrame& df) const = 0;
        virtual double slogl(const DataFrame& df) const = 0;
        virtual BayesianNetworkType type() const = 0;
        virtual DataFrame sample(int n, unsigned int seed) const = 0;
        virtual void save(std::string name, bool include_cpd = false) const = 0;
        virtual std::string ToString() const = 0;
    };

    template<BayesianNetworkType::Value Type>
    class DynamicBayesianNetwork;

    using DynamicGaussianNetwork = DynamicBayesianNetwork<BayesianNetworkType::Gaussian>;
    using DynamicSemiparametricBN = DynamicBayesianNetwork<BayesianNetworkType::Semiparametric>;
    using DynamicDiscreteBN = DynamicBayesianNetwork<BayesianNetworkType::Discrete>;

    template<typename Derived>
    class DynamicBayesianNetworkImpl : public DynamicBayesianNetworkBase {
    public:

        DynamicBayesianNetworkImpl(const std::vector<std::string>& variables, int markovian_order) : m_variables(variables),
                                                                                                     m_markovian_order(markovian_order),
                                                                                                     m_static(),
                                                                                                     m_transition() {
            std::vector<std::string> static_nodes;
            std::vector<std::string> transition_nodes;

            for (const auto& v : variables) {
                transition_nodes.push_back(util::temporal_name(v, 0));
            }

            for (int i = 1; i <= markovian_order; ++i) {
                for (const auto& v : variables) {
                    static_nodes.push_back(util::temporal_name(v, i));
                }
            }

            m_static = BayesianNetwork<BN_traits<Derived>::TYPE>(static_nodes);
            m_transition = ConditionalBayesianNetwork<BN_traits<Derived>::TYPE>(transition_nodes, static_nodes);
        }

        template<typename BN, typename ConditionalBN>
        DynamicBayesianNetworkImpl(const std::vector<std::string>& variables,
                                   int markovian_order,
                                   BN static_bn,
                                   ConditionalBN transition_bn) : m_variables(variables),
                                                                  m_markovian_order(markovian_order),
                                                                  m_static(static_bn),
                                                                  m_transition(transition_bn) {
            
            for (const auto& v : variables) {
                auto present_name = util::temporal_name(v, 0);
                if (!m_transition.contains_node(present_name))
                    throw std::invalid_argument("Node " + present_name + " not present in transition BayesianNetwork.");

                for (int i = 1; i <= m_markovian_order; ++i) {
                    auto name = util::temporal_name(v, i);
                    if (!m_static.contains_node(name))
                        throw std::invalid_argument("Node " + name + " not present in static BayesianNetwork.");
                    if (!m_transition.contains_interface_node(name))
                        throw std::invalid_argument("Interface node " + name + " not present in transition BayesianNetwork.");
                }
            }
        }

        BayesianNetwork<BN_traits<Derived>::TYPE>& static_bn() override { return m_static; }
        const BayesianNetwork<BN_traits<Derived>::TYPE>& static_bn() const override { return m_static; }
        ConditionalBayesianNetwork<BN_traits<Derived>::TYPE>& transition_bn() override { return m_transition; }
        const ConditionalBayesianNetwork<BN_traits<Derived>::TYPE>& transition_bn() const override { return m_transition; }

        int markovian_order() const override {
            return m_markovian_order;
        }

        int num_variables() const override {
            return m_variables.size();
        }

        const std::vector<std::string>& variables() const override {
            return m_variables.elements();
        }

        bool contains_variable(const std::string& name) const override {
            return m_variables.contains(name);
        }

        void add_variable(const std::string& name) override;
        void remove_variable(const std::string& name) override;

        bool fitted() const override {
            return m_static.fitted() && m_transition.fitted();
        }

        void fit(const DataFrame& df) override {
            DynamicDataFrame ddf(df, m_markovian_order);

            m_static.fit(ddf.static_df());
            m_transition.fit(ddf.transition_df());
        }

        void check_fitted() const {
            if (!fitted()) {
                throw std::invalid_argument("DynamicBayesianNetwork currently not fitted. "
                                            "Call fit() method, or add_cpds() for static_bn() and transition_bn()");
            }
        }

        VectorXd logl(const DataFrame& df) const override;
        double slogl(const DataFrame& df) const override;
        
        BayesianNetworkType type() const override {
            return BN_traits<Derived>::TYPE;
        }

        DataFrame sample(int n, unsigned int seed) const override;
        void save(std::string name, bool include_cpd = false) const override;

        py::tuple __getstate__() const;
        static Derived __setstate__(py::tuple& t);
    private:
        BidirectionalMapIndex<std::string> m_variables;
        int m_markovian_order;
        BayesianNetwork<BN_traits<Derived>::TYPE> m_static;
        ConditionalBayesianNetwork<BN_traits<Derived>::TYPE> m_transition;
        mutable bool m_include_cpd;
    };
    
    template<typename BN, typename ConditionalBN>
    DynamicBayesianNetworkImpl(BN static_bn, ConditionalBN transition_bn) 
        -> DynamicBayesianNetworkImpl<DynamicBayesianNetwork<BN::TYPE>>;

    template<typename Derived>
    void DynamicBayesianNetworkImpl<Derived>::add_variable(const std::string& name) {
        if (contains_variable(name)) {
            throw std::invalid_argument("Cannot add variable " + name + " because a variable with the same name already exists.");
        }

        m_variables.insert(name);

        m_transition.add_node(util::temporal_name(name, 0));

        for (int i = 1; i <= m_markovian_order; ++i) {
            auto new_name = util::temporal_name(name, i);
            m_static.add_node(new_name);
            m_transition.add_interface_node(new_name);
        }
    }

    template<typename Derived>
    void DynamicBayesianNetworkImpl<Derived>::remove_variable(const std::string& name) {
        if (!contains_variable(name)) {
            throw std::invalid_argument("Cannot remove variable " + name + " because a variable with the same name do not exist.");
        }

        m_variables.remove(name);

        m_transition.remove_node(util::temporal_name(name, 0));

        for (int i = 1; i <= m_markovian_order; ++i) {
            auto new_name = util::temporal_name(name, i);
            m_static.remove_node(new_name);
            m_transition.remove_interface_node(new_name);
        }
    }

    template<typename Derived>
    VectorXd DynamicBayesianNetworkImpl<Derived>::logl(const DataFrame& df) const {
        check_fitted();

        if (df->num_rows() < m_markovian_order)
            throw std::invalid_argument("Not enough information. There are less rows in "
                                        "test DataFrame (" + std::to_string(df->num_rows()) + ")"
                                        " than the markovian order of the "
                                        "DynamicBayesianNetwork (" + std::to_string(m_markovian_order) + ")");


        VectorXd ll = VectorXd::Zero(df->num_rows());

        auto static_df = df.slice(0, m_markovian_order);
        auto dstatic_df = create_static_df(static_df, m_markovian_order);

        // Generate logl for the static BN.
        for (int i = 0; i < m_markovian_order; ++i) {
            for (const auto& v : m_variables) {
                const auto& cpd = m_static.cpd(util::temporal_name(v, m_markovian_order-i));
                ll(i) += cpd.slogl(dstatic_df);
            }
        }

        auto temporal_slices = create_temporal_slices(df, m_markovian_order);
        auto dtransition_df = create_transition_df(temporal_slices, m_markovian_order);

        // Generate logl for the transition BN
        for (const auto& v : m_variables) {
            auto name = util::temporal_name(v, 0);

            const auto& cpd = m_transition.cpd(name);
            auto vll = cpd.logl(dtransition_df);

            for (int i = 0; i < vll.rows(); ++i) {
                ll(i+m_markovian_order) += vll(i);
            }
        }

        return ll;
    }

    template<typename Derived>
    double DynamicBayesianNetworkImpl<Derived>::slogl(const DataFrame& df) const {
        check_fitted();

        if (df->num_rows() < m_markovian_order)
            throw std::invalid_argument("Not enough information. There are less rows in "
                                        "test DataFrame (" + std::to_string(df->num_rows()) + ")"
                                        " than the markovian order of the "
                                        "DynamicBayesianNetwork (" + std::to_string(m_markovian_order) + ")");

        double sll = 0;
        
        auto static_df = df.slice(0, m_markovian_order);
        auto dstatic_df = create_static_df(static_df, m_markovian_order);

        // Generate slogl for the static BN.
        for (int i = 0; i < m_markovian_order; ++i) {
            for (const auto& v : m_variables) {
                const auto& cpd = m_static.cpd(util::temporal_name(v, m_markovian_order-i));
                sll += cpd.slogl(dstatic_df);
            }
        }

        auto temporal_slices = create_temporal_slices(df, m_markovian_order);
        auto dtransition_df = create_transition_df(temporal_slices, m_markovian_order);

        // Generate logl for the transition BN
        for (const auto& v : m_variables) {
            auto name = util::temporal_name(v, 0);
            const auto& cpd = m_transition.cpd(name);
            sll += cpd.slogl(dtransition_df);
        }

        return sll;
    }

    Array_ptr new_numeric_array(arrow::Type::type t, int length);
    Array_ptr new_discrete_array(const std::vector<std::string>& values, int length);

    template<typename Derived>
    arrow::Type::type find_type(const DynamicBayesianNetworkImpl<Derived>& dbn, const std::string& variable) {

        if constexpr (BN_traits<Derived>::TYPE == BayesianNetworkType::Gaussian) {
            return Type::DOUBLE;
        } else if constexpr (BN_traits<Derived>::TYPE == BayesianNetworkType::Discrete) {
            return Type::DICTIONARY;
        } else {
            arrow::Type::type t = arrow::Type::NA;

            for (int i = 1; i <= dbn.markovian_order(); ++i) {
                const auto& cpd = dbn.static_bn().cpd(util::temporal_name(variable, i));

                switch (cpd.factor_type()) {
                    case FactorType::LinearGaussianCPD:
                        return Type::DOUBLE;
                    case FactorType::CKDE: {
                        const auto& ckde = cpd.as_ckde();
                        if (ckde.arrow_type() == Type::DOUBLE)
                            return Type::DOUBLE;
                        else
                            t = ckde.arrow_type();
                        break;
                    }
                    default:
                        throw std::runtime_error("Unreachable code.");
                }
            }


            const auto& cpd = dbn.transition_bn().cpd(util::temporal_name(variable, 0));
            switch (cpd.factor_type()) {
                case FactorType::LinearGaussianCPD:
                    return Type::DOUBLE;
                case FactorType::CKDE: {
                    const auto& ckde = cpd.as_ckde();
                    if (ckde.arrow_type() == Type::DOUBLE)
                        return Type::DOUBLE;
                    else
                        t = ckde.arrow_type();
                    break;
                }
                default:
                    throw std::runtime_error("Unreachable code.");
            }

            return t;
        }
    }

    template<typename Derived>
    std::vector<std::string> discretefactor_possible_values(const DynamicBayesianNetworkImpl<Derived>& dbn,
                                                            const std::string& variable) {
        std::unordered_set<std::string> values;

        for (int i = 1; i < dbn.markovian_order(); ++i) {
            const auto& cpd = dbn.static_bn().cpd(util::temporal_name(variable, i));

            for (const auto& value : cpd.variable_values()) {
                values.insert(value);
            }
        }

        const auto& cpd = dbn.transition_bn().cpd(util::temporal_name(variable, 0));

        for (const auto& value : cpd.variable_values()) {
            values.insert(value);
        }

        return std::vector(values.begin(), values.end());
    }

    template<typename Derived>
    std::pair<DataFrame, std::unordered_map<std::string, arrow::Type::type>>
    generate_empty_dataframe(const DynamicBayesianNetworkImpl<Derived>& dbn, int n) {
        
        std::unordered_map<std::string, arrow::Type::type> types;
        std::vector<Array_ptr> columns;
        std::vector<Field_ptr> fields;

        for (const auto& variable : dbn.variables()) {
            if constexpr (BN_traits<Derived>::TYPE == BayesianNetworkType::Discrete) {
                types.insert({variable, Type::DICTIONARY});
                auto variable_values = discretefactor_possible_values(dbn, variable);
                columns.push_back(new_discrete_array(variable_values, n));
            } else {
                auto type = find_type(dbn, variable);
                types.insert({variable, type});
                columns.push_back(new_numeric_array(type, n));
            }

            fields.push_back(arrow::field(variable, columns.back()->type()));
        }

        auto schema = arrow::schema(fields);

        return {arrow::RecordBatch::Make(schema, n, columns), types};
    }

    template<typename ArrowType, typename Derived>
    void sample_discrete_static_bn(const std::string& variable,
                                   int max_length,
                                   const DataFrame& static_sample,
                                   const DynamicBayesianNetworkImpl<Derived>& dbn,
                                   Array_ptr& output_indices) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto raw_output = output_indices->data()->template GetMutableValues<typename ArrowType::c_type>(1);
        for (int i = 0; i < max_length; ++i) {
            auto name = util::temporal_name(variable, dbn.markovian_order()-i);
            
            auto input = static_sample.col(name);
            auto dict_input = std::static_pointer_cast<arrow::DictionaryArray>(input);
            auto input_indices = dict_input->indices();

            auto dwn_input_indices = std::static_pointer_cast<ArrayType>(input_indices);
            raw_output[i] = dwn_input_indices->Value(0);
        }
    }

    template<typename Derived>
    void sample_static_bn(DataFrame& df,
                          const std::unordered_map<std::string, arrow::Type::type>& types,
                          const DynamicBayesianNetworkImpl<Derived>& dbn,
                          int n, 
                          unsigned int seed) {
        auto static_sample = dbn.static_bn().sample(1, seed);

        auto max_length = std::min(dbn.markovian_order(), n);

        for (const auto& v : dbn.variables()) {
            switch (types.at(v)) {
                case Type::DOUBLE: {
                    auto raw_col = df.template mutable_data<arrow::DoubleType>(v);

                    for (int i = 0; i < max_length; ++i) {
                        auto name = util::temporal_name(v, dbn.markovian_order()-i);
                        raw_col[i] = *static_sample.template data<arrow::DoubleType>(name);
                    }
                    break;
                }
                case Type::FLOAT: {
                    auto raw_col = df.mutable_data<arrow::FloatType>(v);

                    for (int i = 0; i < max_length; ++i) {
                        auto name = util::temporal_name(v, dbn.markovian_order()-i);
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
                    throw std::runtime_error("Wrong data type for variable " + v + ": " + std::to_string(types.at(v)));
            }
        }
    }

    std::vector<std::string> conditional_topological_sort(const ConditionalDag& dag);

    template<typename Derived>
    void sample_transition_bn(DataFrame& df,
                              const std::unordered_map<std::string, arrow::Type::type>& types,
                              const DynamicBayesianNetworkImpl<Derived>& dbn,
                              int n, 
                              unsigned int seed) {
        
        auto top_sort = conditional_topological_sort(dbn.transition_bn().graph());

        for (int i = dbn.markovian_order(); i < n; ++i) {
            auto evidence_slice = df.slice(i-dbn.markovian_order(), dbn.markovian_order()+1);
            DynamicDataFrame slice_ddf(evidence_slice, dbn.markovian_order());
            for (const auto& full_variable : top_sort) {
                // Remove the suffix "_t_0"
                auto variable = full_variable.substr(0, full_variable.size()-4);
                const auto& cpd = dbn.transition_bn().cpd(full_variable);

                auto sampled = cpd.sample(1, slice_ddf.transition_df(), seed);

                switch (types.at(variable)) {
                    case Type::DOUBLE: {
                        auto raw_col = df.template mutable_data<arrow::DoubleType>(variable);
                        
                        switch(sampled->type_id()) {
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
                        
                        switch(sampled->type_id()) {
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
                                throw std::invalid_argument("Wrong indices array type of DictionaryArray "
                                                            "for variable " + variable);
                            }

                        break;
                    }
                    default:
                        throw std::runtime_error("Wrong data type for variable " + variable +
                                                 ": " + std::to_string(types.at(variable)));
                }
            }
        }
    }

    template<typename Derived>
    DataFrame DynamicBayesianNetworkImpl<Derived>::sample(int n, unsigned int seed) const {
        check_fitted();

        auto [sampled, types] = generate_empty_dataframe(*this, n);

        sample_static_bn(sampled, types, *this, n, seed);
        sample_transition_bn(sampled, types, *this, n, seed);

        return sampled;
    }

    template<typename Derived>
    py::tuple DynamicBayesianNetworkImpl<Derived>::__getstate__() const {
        m_static.m_include_cpd = m_include_cpd;
        m_transition.m_include_cpd = m_include_cpd;
        return py::make_tuple(m_variables.elements(),
                              m_markovian_order,
                              m_static.__getstate__(),
                              m_transition.__getstate__());
    }

    template<typename Derived>
    Derived DynamicBayesianNetworkImpl<Derived>::__setstate__(py::tuple& t) {
        if (t.size() != 4)
            throw std::runtime_error("Not valid DynamicBayesianNetwork");

        auto variables = t[0].cast<std::vector<std::string>>();
        auto markovian_order = t[1].cast<int>();
        auto static_bn = t[2].cast<BayesianNetwork<BN_traits<Derived>::TYPE>>();
        auto transition_bn = t[3].cast<ConditionalBayesianNetwork<BN_traits<Derived>::TYPE>>();

        return Derived(variables, markovian_order, static_bn, transition_bn);
    }

    template<typename Derived>
    void DynamicBayesianNetworkImpl<Derived>::save(std::string name, bool include_cpd) const {
        m_include_cpd = include_cpd;
        auto open = py::module::import("io").attr("open");

        if (name.size() < 7 || name.substr(name.size()-7) != ".pickle")
            name += ".pickle";
            
        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(static_cast<const Derived*>(this)), file, 2);
        file.attr("close")();
    }
}

#endif //PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP