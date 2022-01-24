#ifndef PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP

#include <dataset/dynamic_dataset.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <util/temporal.hpp>

using dataset::DynamicDataFrame;
using factors::continuous::LinearGaussianCPD, factors::continuous::CKDE, factors::discrete::DiscreteFactor;
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
    virtual void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) = 0;
    virtual VectorXd logl(const DataFrame& df) const = 0;
    virtual double slogl(const DataFrame& df) const = 0;
    virtual std::shared_ptr<BayesianNetworkType> type() const = 0;
    virtual BayesianNetworkType& type_ref() const = 0;
    virtual DataFrame sample(int n, unsigned int seed = std::random_device{}()) const = 0;
    virtual py::tuple __getstate__() const = 0;
    virtual void save(std::string name, bool include_cpd = false) const = 0;
    virtual std::string ToString() const = 0;
};

class DynamicBayesianNetwork : public clone_inherit<DynamicBayesianNetwork, DynamicBayesianNetworkBase> {
public:
    DynamicBayesianNetwork(std::shared_ptr<BayesianNetworkType> type,
                           const std::vector<std::string>& variables,
                           int markovian_order)
        : m_variables(variables), m_markovian_order(markovian_order), m_static(), m_transition() {
        if (type == nullptr) throw std::runtime_error("Type of Bayesian network must be non-null.");

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

        m_static = type->new_bn(static_nodes);
        m_transition = type->new_cbn(transition_nodes, static_nodes);
    }

    DynamicBayesianNetwork(const std::vector<std::string>& variables,
                           int markovian_order,
                           std::shared_ptr<BayesianNetworkBase> static_bn,
                           std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : m_variables(variables), m_markovian_order(markovian_order), m_static(static_bn), m_transition(transition_bn) {
        if (static_bn == nullptr) throw std::runtime_error("Static Bayesian network must be non-null.");

        if (transition_bn == nullptr) throw std::runtime_error("Transition Bayesian network must be non-null.");

        if (static_bn->type_ref() != transition_bn->type_ref())
            throw std::invalid_argument("Static and transition Bayesian networks do not have the same type.");

        for (const auto& v : variables) {
            auto present_name = util::temporal_name(v, 0);
            if (!m_transition->contains_node(present_name))
                throw std::invalid_argument("Node " + present_name + " not present in transition BayesianNetwork.");

            for (int i = 1; i <= m_markovian_order; ++i) {
                auto name = util::temporal_name(v, i);
                if (!m_static->contains_node(name))
                    throw std::invalid_argument("Node " + name + " not present in static BayesianNetwork.");
                if (!m_transition->contains_interface_node(name))
                    throw std::invalid_argument("Interface node " + name +
                                                " not present in transition BayesianNetwork.");
            }
        }
    }

    BayesianNetworkBase& static_bn() override { return *m_static; }
    const BayesianNetworkBase& static_bn() const override { return *m_static; }
    ConditionalBayesianNetworkBase& transition_bn() override { return *m_transition; }
    const ConditionalBayesianNetworkBase& transition_bn() const override { return *m_transition; }

    int markovian_order() const override { return m_markovian_order; }

    int num_variables() const override { return m_variables.size(); }

    const std::vector<std::string>& variables() const override { return m_variables.elements(); }

    bool contains_variable(const std::string& name) const override { return m_variables.contains(name); }

    void add_variable(const std::string& name) override;
    void remove_variable(const std::string& name) override;

    bool fitted() const override { return m_static->fitted() && m_transition->fitted(); }

    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override {
        DynamicDataFrame ddf(df, m_markovian_order);

        m_static->fit(ddf.static_df(), construction_args);
        m_transition->fit(ddf.transition_df(), construction_args);
    }

    void check_fitted() const {
        if (!fitted()) {
            throw std::invalid_argument(
                "DynamicBayesianNetwork currently not fitted. "
                "Call fit() method, or add_cpds() for static_bn() and transition_bn()");
        }
    }

    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    std::shared_ptr<BayesianNetworkType> type() const override { return m_transition->type(); }

    BayesianNetworkType& type_ref() const override { return m_transition->type_ref(); }

    DataFrame sample(int n, unsigned int seed) const override;

    std::string ToString() const override { return "Dynamic" + type_ref().ToString(); }

    void save(std::string name, bool include_cpd = false) const override;

    py::tuple __getstate__() const override;

    bool include_cpd() const { return m_include_cpd; }

    void set_include_cpd(bool include_cpd) const { m_include_cpd = include_cpd; }

private:
    std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> check_same_datatypes() const;

    BidirectionalMapIndex<std::string> m_variables;
    int m_markovian_order;
    std::shared_ptr<BayesianNetworkBase> m_static;
    std::shared_ptr<ConditionalBayesianNetworkBase> m_transition;
    mutable bool m_include_cpd;
};

void __nonderived_dbn_setstate__(py::object& self, py::tuple& t);

template <typename DerivedBN>
std::shared_ptr<DerivedBN> __derived_dbn_setstate__(py::tuple& t) {
    if (t.size() != 4) throw std::runtime_error("Not valid DynamicBayesianNetwork");

    auto variables = t[0].cast<std::vector<std::string>>();
    auto markovian_order = t[1].cast<int>();
    // These BNs are C++ BNs, so no need to keep the Python objects alive.
    auto static_bn = t[2].cast<std::shared_ptr<BayesianNetworkBase>>();
    auto transition_bn = t[3].cast<std::shared_ptr<ConditionalBayesianNetworkBase>>();

    return std::make_shared<DerivedBN>(variables, markovian_order, static_bn, transition_bn);
}

}  // namespace models

#endif  // PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP