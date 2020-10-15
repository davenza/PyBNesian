#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <util/util_types.hpp>

using dataset::DataFrame;
using graph::DirectedGraph;

using factors::continuous::LinearGaussianCPD;
using factors::continuous::SemiparametricCPD;

using util::ArcVector, util::FactorTypeVector;

using Field_ptr = std::shared_ptr<arrow::Field>;
using Array_ptr = std::shared_ptr<arrow::Array>;

namespace models {

    template<typename Model>
    struct BN_traits {};

    class BayesianNetworkType
    {
    public:
        enum Value : uint8_t
        {
            GBN,
            DISCRETEBN,
            SPBN
        };

        struct Hash
        {
            inline std::size_t operator ()(BayesianNetworkType const bn_type) const
            {
                return static_cast<std::size_t>(bn_type.value);
            }
        };

        using HashType = Hash;

        BayesianNetworkType() = default;
        constexpr BayesianNetworkType(Value bn_type) : value(bn_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(BayesianNetworkType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(BayesianNetworkType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::GBN:
                    return "GaussianNetwork";
                case Value::SPBN:
                    return "SemiparametricBN";
                default:
                    throw std::invalid_argument("Unreachable code in BayesianNetworkType.");
            }
        }

    private:
        Value value;
    };

    class BayesianNetworkBase {
    public:
        virtual ~BayesianNetworkBase() = default;
        virtual int num_nodes() const = 0;
        virtual int num_arcs() const = 0;
        virtual std::vector<std::string> nodes() const = 0;
        virtual ArcVector arcs() const = 0;
        virtual const std::unordered_map<std::string, int>& indices() const = 0;
        virtual bool contains_node(const std::string& name) const = 0;
        virtual const std::string& name(int node_index) const = 0;
        virtual int num_parents(int node_index) const = 0;
        virtual int num_parents(const std::string& node) const = 0;
        virtual int num_children(int node_index) const = 0;
        virtual int num_children(const std::string& node) const = 0;
        virtual int index(const std::string& node) const = 0;
        virtual std::vector<std::string> parents(int node_index) const = 0;
        virtual std::vector<std::string> parents(const std::string& node) const = 0;
        virtual std::vector<int> parent_indices(int node_index) const = 0;
        virtual std::vector<int> parent_indices(const std::string& node) const = 0;
        virtual std::string parents_to_string(int node_index) const = 0;
        virtual std::string parents_to_string(const std::string& node) const = 0;
        virtual bool has_arc(int source, int dest) const = 0;
        virtual bool has_arc(const std::string& source, const std::string& dest) const = 0;
        virtual bool has_path(int source_index, int dest_index) const = 0;
        virtual bool has_path(const std::string& source, const std::string& dest) const = 0;
        virtual void add_arc(int source, int dest) = 0;
        virtual void add_arc(const std::string& source, const std::string& dest) = 0;
        virtual bool can_add_arc(int source_index, int dest_index) const = 0;
        virtual bool can_add_arc(const std::string& source, const std::string& dest) const = 0;
        virtual bool can_flip_arc(int source_index, int dest_index) = 0;
        virtual bool can_flip_arc(const std::string& source, const std::string& dest) = 0;
        virtual void remove_arc(int source, int dest) = 0;
        virtual void remove_arc(const std::string& source, const std::string& dest) = 0;
        virtual void fit(const DataFrame& df) = 0;
        virtual VectorXd logl(const DataFrame& df) const = 0;
        virtual double slogl(const DataFrame& df) const = 0;
        virtual std::string ToString() const = 0;
        virtual BayesianNetworkType type() const = 0;
        virtual DataFrame sample(int n, long unsigned int seed, bool ordered) const = 0;
    };

    class SemiparametricBNBase {
    public:
        virtual FactorType node_type(int node_index) const = 0;
        virtual FactorType node_type(const std::string& node) const = 0;
        virtual void set_node_type(int node_index, FactorType new_type) = 0;
        virtual void set_node_type(const std::string& node, FactorType new_type) = 0;
    };


    template<typename Derived>
    class BayesianNetwork : public BayesianNetworkBase {
    public:
        // using DagType = typename BN_traits<Derived>::DagType;
        using CPD = typename BN_traits<Derived>::CPD;

        BayesianNetwork(const std::vector<std::string>& nodes);
        BayesianNetwork(const ArcVector& arcs);
        BayesianNetwork(const std::vector<std::string>& nodes, const ArcVector& arcs);

        int num_nodes() const override {
            return g.num_nodes();
        }

        int num_arcs() const override {
            return g.num_arcs();
        }

        std::vector<std::string> nodes() const override {
            return g.nodes();
        }

        ArcVector arcs() const override {
            return g.arcs();
        }

        const std::unordered_map<std::string, int>& indices() const override {
            return g.indices();
        }

        bool contains_node(const std::string& name) const override {
            return g.contains_node(name);
        }

        const std::string& name(int node_index) const override {
            return g.name(node_index);
        }

        int num_parents(int node_index) const override {
            return g.num_parents(node_index);
        }

        int num_parents(const std::string& node) const override {
            return g.num_parents(node);
        }

        int num_children(int node_index) const override {
            return g.num_children(node_index);
        }

        int num_children(const std::string& node) const override {
            return g.num_children(node);
        }

        int index(const std::string& node) const override {
            return g.index(node);
        }

        std::vector<std::string> parents(int node_index) const override {
            return g.parents(node_index);
        }

        std::vector<std::string> parents(const std::string& node) const override {
            return g.parents(node);
        }

        std::vector<int> parent_indices(int node_index) const override {
            return g.parent_indices(node_index);
        }

        std::vector<int> parent_indices(const std::string& node) const override {
            return g.parent_indices(node);
        }

        std::string parents_to_string(int node_index) const override {
            return g.parents_to_string(node_index);
        }

        std::string parents_to_string(const std::string& node) const override {
            return g.parents_to_string(node);
        }

        bool has_arc(int source, int target) const override {
            return g.has_arc(source, target);
        }

        bool has_arc(const std::string& source, const std::string& target) const override {
            return g.has_arc(source, target);
        }

        bool has_path(int source_index, int target_index) const override {
            return g.has_path(source_index, target_index);
        }
        
        bool has_path(const std::string& source, const std::string& target) const override {
            return g.has_path(source, target);
        }

        void add_arc(int source, int target) override {
            g.add_arc(source, target);
        }

        void add_arc(const std::string& source, const std::string& target) override {
            g.add_arc(source, target);
        }

        bool can_add_arc(int source_index, int target_index) const override {
            return g.can_add_arc(source_index, target_index);
        }

        bool can_add_arc(const std::string& source, const std::string& target) const override {
            return g.can_add_arc(source, target);
        }

        bool can_flip_arc(int source_index, int target_index) override {
            return g.can_flip_arc(source_index, target_index);
        }

        bool can_flip_arc(const std::string& source, const std::string& target) override {
            return g.can_flip_arc(source, target);
        }

        void remove_arc(int source, int target) override {
            g.remove_arc(source, target);
        }

        void remove_arc(const std::string& source, const std::string& target) override {
            g.remove_arc(source, target);
        }

        void check_blacklist(const ArcVector& arc_blacklist) const {
            for(auto& arc : arc_blacklist) {
                if (has_arc(arc.first, arc.second)) {
                    throw std::invalid_argument("Edge " + arc.first + " -> " + arc.second + " in blacklist,"
                                                " but it is present in the Bayesian Network.");
                }
            }
        }

        void force_whitelist(const ArcVector& arc_whitelist) {
            for(auto& arc : arc_whitelist) {
                if (!has_arc(arc.first, arc.second)) {
                    if (has_arc(arc.second, arc.first)) {
                        throw std::invalid_argument("Edge " + arc.first + " -> " + arc.second + " in whitelist,"
                                                    " but edge " + arc.second + " -> " + arc.first + " is present"
                                                    " in the Bayesian Network.");
                    } else {
                        add_arc(arc.first, arc.second);
                    }
                }
            }
        }

        void force_type_whitelist(const FactorTypeVector&) {}

        void add_cpds(const std::vector<CPD>& cpds);

        void compatible_cpd(const CPD& cpd) const;
        
        void fit(const DataFrame& df) override;
        bool must_refit_cpd(const CPD& node) const;

        CPD create_cpd(const std::string& node) {
            auto pa = parents(node);
            return CPD(node, pa);
        }

        CPD& cpd(int index) {
            if (!m_cpds.empty())
                return m_cpds[index];
            else
                throw py::value_error("CPD of variable \"" + name(index) + "\" not added. Call add_cpds() or fit() to add the CPD.");
        }

        CPD& cpd(const std::string& node) {
            return cpd(g.index(node));
        }

        VectorXd logl(const DataFrame& df) const override;
        double slogl(const DataFrame& df) const override;

        DataFrame sample(int n, long unsigned int seed = std::random_device{}(), bool ordered = false) const override;

        template<typename Derived_>
        friend std::ostream& operator<<(std::ostream &os, const BayesianNetwork<Derived_>& bn);

    protected:
        void check_fitted() const;
    private:
        DirectedGraph g;
        std::vector<CPD> m_cpds;
    };

    template<typename Derived_>
    std::ostream& operator<<(std::ostream &os, const BayesianNetwork<Derived_>& bn) {
        os << "Bayesian network: " << std::endl;
        for(auto& [source, target] : bn.g.arcs())
            os << source << " -> " << target << std::endl;
        return os;
    }

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes), m_cpds() {};

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const ArcVector& arcs) : g(arcs), m_cpds() {};

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const std::vector<std::string>& nodes, 
                                              const ArcVector& arcs) 
                                                 : g(nodes, arcs), m_cpds() {};
    template<typename Derived>
    void BayesianNetwork<Derived>::compatible_cpd(const CPD& cpd) const {
        if (!contains_node(cpd.variable())) {
            throw std::invalid_argument("CPD defined on variable which is not present in the model:\n" + cpd.ToString());
        }

        auto& evidence = cpd.evidence();

        for (auto& ev : evidence) {
            if (!contains_node(ev)) {
                throw std::invalid_argument("Evidence variable " + ev + " is not present in the model:\n" + cpd.ToString());
            }
        }

        auto pa = parents(cpd.variable());
        if (pa.size() != evidence.size()) {
            std::string err = "CPD do not have the model's parent set as evidence:\n" + cpd.ToString() 
                                + "\nParents: " + parents_to_string(cpd.variable());

            throw std::invalid_argument(err);
        }

        std::unordered_set<std::string> evidence_set(evidence.begin(), evidence.end());
        for (auto& parent : pa) {
            if (evidence_set.find(parent) == evidence_set.end()) {
                std::string err = "CPD do not have the model's parent set as evidence:\n" + cpd.ToString() 
                                    + "\nParents: [";
                throw std::invalid_argument(err);
            }
        }
    }

    template<typename Derived>
    void BayesianNetwork<Derived>::add_cpds(const std::vector<CPD>& cpds) {
        
        for (auto& cpd : cpds) {
            static_cast<Derived*>(this)->compatible_cpd(cpd);
        }

        if (m_cpds.empty()) {
            std::unordered_map<std::string, typename std::vector<CPD>::const_iterator> map_index;
            for (auto it = cpds.begin(); it != cpds.end(); ++it) {
                if (map_index.count(it->variable()) == 1) {
                    throw std::invalid_argument("CPD for variable " + it->variable() + "is repeated.");
                }
                map_index[it->variable()] = it;
            }
            m_cpds.reserve(num_nodes());
            for(auto& node : nodes()) {
                auto cpd_idx = map_index.find(node);

                if (cpd_idx != map_index.end()) {
                    auto cpd = *(cpd_idx->second);
                    m_cpds.push_back(cpd);
                } else {
                    m_cpds.push_back(static_cast<Derived*>(this)->create_cpd(node));
                }
            }
        } else {
            for(auto& cpd : cpds) {
                auto idx = index(cpd.variable());
                m_cpds[idx] = cpd;
            }
        }
    }

    template<typename Derived>
    void BayesianNetwork<Derived>::fit(const DataFrame& df) {
        if (m_cpds.empty()) {
            m_cpds.reserve(g.num_nodes());

            for (auto& node : g.nodes()) {
                auto cpd = static_cast<Derived*>(this)->create_cpd(node);
                m_cpds.push_back(cpd);
                m_cpds.back().fit(df);
            }
        } else {
            for (auto& cpd : m_cpds) {
                if (static_cast<Derived*>(this)->must_refit_cpd(cpd)) {
                    cpd = static_cast<Derived*>(this)->create_cpd(cpd.variable());
                    cpd.fit(df);
                } else if (!cpd.fitted()) {
                    cpd.fit(df);
                }
            }
        }
    }

    template<typename Derived>
    bool BayesianNetwork<Derived>::must_refit_cpd(const CPD& cpd) const {
        auto& node = cpd.variable();
        auto& cpd_evidence = cpd.evidence();
        auto parents = this->parents(node);
        if (cpd_evidence.size() != parents.size())
            return true;

        if (std::is_permutation(cpd_evidence.begin(), cpd_evidence.end(), parents.begin(), parents.end())) {
            return false;
        } else {
            return true;
        }
    }

    template<typename Derived>
    void BayesianNetwork<Derived>::check_fitted() const {
        if (m_cpds.empty()) {
            py::value_error("Model not fitted.");
        } else {
            bool all_fitted = true;
            std::string err;
            for (auto& cpd : m_cpds) {
                if (!cpd.fitted()) {
                    if (all_fitted) {
                        err += "Some CPDs are not fitted:\n";
                        all_fitted = false;
                    }
                    err += cpd.ToString() + "\n";
                }
            }
            if (!all_fitted)
                throw py::value_error(err);
        }
    }

    template<typename Derived>
    VectorXd BayesianNetwork<Derived>::logl(const DataFrame& df) const {
        check_fitted();

        VectorXd accum = m_cpds[0].logl(df);
        for (auto it = ++m_cpds.begin(); it != m_cpds.end(); ++it) {
            accum += it->logl(df);
        }
        return accum;
    }

    template<typename Derived>
    double BayesianNetwork<Derived>::slogl(const DataFrame& df) const {
        check_fitted();
        
        double accum = m_cpds[0].slogl(df);
        for (auto it = ++m_cpds.begin(); it != m_cpds.end(); ++it) {
            accum += it->slogl(df);
        }
        return accum;
    }

    template<typename Derived>
    DataFrame BayesianNetwork<Derived>::sample(int n, long unsigned int seed, bool ordered) const {
        check_fitted();

        DataFrame parents(n);

        int i = 0;
        for (auto& name : g.topological_sort()) {
            auto index = g.index(name);
            auto array = m_cpds[index].sample(n, parents, seed);
            
            auto res = parents->AddColumn(i, name, array);
            parents = DataFrame(std::move(res).ValueOrDie());
            ++i;
        }

        if (ordered) {
            std::vector<Field_ptr> fields;
            std::vector<Array_ptr> columns;
            
            auto schema = parents->schema();
            for (auto& name : nodes()) {
                fields.push_back(schema->GetFieldByName(name));
                columns.push_back(parents->GetColumnByName(name));
            }

            auto new_schema = std::make_shared<arrow::Schema>(fields);

            auto new_rb = arrow::RecordBatch::Make(new_schema, n, columns);
            return DataFrame(new_rb);
        } else {
            return parents;
        }
    }

    void requires_continuous_data(const DataFrame& df);
    void requires_discrete_data(const DataFrame& df);
}

#endif //PGM_DATASET_BAYESIANNETWORK_HPP