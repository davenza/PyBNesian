#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <util/util_types.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag, graph::AdjListDag;
using boost::source;

using factors::continuous::LinearGaussianCPD;
using factors::continuous::SemiparametricCPD;

using util::ArcVector, util::FactorTypeVector;

namespace models {

    template<typename Model>
    struct BN_traits {};

    class BayesianNetworkType
    {
    public:
        enum Value : uint8_t
        {
            GBN,
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
        virtual int num_nodes() const = 0;
        virtual int num_edges() const = 0;
        virtual const std::vector<std::string>& nodes() const = 0;
        virtual ArcVector edges() const = 0;
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
        virtual std::string parents_tostring(int node_index) const = 0;
        virtual std::string parents_tostring(const std::string& node) const = 0;
        virtual bool has_edge(int source, int dest) const = 0;
        virtual bool has_edge(const std::string& source, const std::string& dest) const = 0;
        virtual bool has_path(int source_index, int dest_index) const = 0;
        virtual bool has_path(const std::string& source, const std::string& dest) const = 0;
        virtual void add_edge(int source, int dest) = 0;
        virtual void add_edge(const std::string& source, const std::string& dest) = 0;
        virtual bool can_add_edge(int source_index, int dest_index) const = 0;
        virtual bool can_add_edge(const std::string& source, const std::string& dest) const = 0;
        virtual bool can_flip_edge(int source_index, int dest_index) = 0;
        virtual bool can_flip_edge(const std::string& source, const std::string& dest) = 0;
        virtual void remove_edge(int source, int dest) = 0;
        virtual void remove_edge(const std::string& source, const std::string& dest) = 0;
        virtual void fit(const DataFrame& df) = 0;
        virtual VectorXd logpdf(const DataFrame& df) const = 0;
        virtual double slogpdf(const DataFrame& df) const = 0;
        virtual std::string ToString() const = 0;
        virtual BayesianNetworkType type() const = 0;
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
        using DagType = typename BN_traits<Derived>::DagType;
        using CPD = typename BN_traits<Derived>::CPD;
        using node_descriptor = typename DagType::node_descriptor;
        using edge_descriptor = typename DagType::edge_descriptor;

        using node_iterator_t = typename DagType::node_iterator_t;

        BayesianNetwork(const std::vector<std::string>& nodes);
        BayesianNetwork(const ArcVector& arcs);
        BayesianNetwork(const std::vector<std::string>& nodes, const ArcVector& arcs);

        int num_nodes() const override {
            return g.num_nodes();
        }

        int num_edges() const override {
            return g.num_edges();
        }

        const std::vector<std::string>& nodes() const override {
            return m_nodes;
        }

        ArcVector edges() const override {
            ArcVector res;
            res.reserve(num_edges());

            for (auto [eit, eend] = g.edges(); eit != eend; ++eit) {
                res.push_back(std::make_pair(name(g.source(*eit)), name(g.target(*eit))));
            }

            return res;
        }

        const std::unordered_map<std::string, int>& indices() const override {
            return m_indices;
        }

        node_descriptor node(const std::string& node) const {
            return g.node(m_indices.at(node));
        }

        node_descriptor node(int node_index) const {
            return g.node(node_index);
        }

        bool contains_node(const std::string& name) const override {
            return m_indices.count(name) > 0;
        }

        const std::string& name(int node_index) const override {
            return m_nodes[node_index];
        }

        const std::string& name(node_descriptor node) const {
            return name(g.index(node));
        }

        int num_parents(node_descriptor node) const {
            return g.num_parents(node);
        }

        int num_parents(int node_index) const override {
            return num_parents(g.node(node_index));
        }

        int num_parents(const std::string& node) const override {
            return num_parents(m_indices.at(node));
        }

        int num_children(node_descriptor node) const {
            return g.num_children(node);
        }

        int num_children(int node_index) const override {
            return num_children(g.node(node_index));
        }

        int num_children(const std::string& node) const override {
            return num_children(m_indices.at(node));
        }

        int index(node_descriptor n) const {
            return g.index(n);
        }

        int index(const std::string& node) const override {
            return m_indices.at(node);
        }

        std::vector<std::string> parents(node_descriptor node) const;

        std::vector<std::string> parents(int node_index) const override {
            return parents(g.node(node_index));
        }

        std::vector<std::string> parents(const std::string& node) const override {
            return parents(m_indices.at(node));
        }

        std::vector<int> parent_indices(node_descriptor node) const;

        std::vector<int> parent_indices(int node_index) const override {
            return parent_indices(g.node(node_index));
        }

        std::vector<int> parent_indices(const std::string& node) const override {
            return parent_indices(m_indices.at(node));
        }

        std::string parents_tostring(node_descriptor node) const;

        std::string parents_tostring(int node_index) const override {
            return parents_tostring(g.node(node_index));
        }

        std::string parents_tostring(const std::string& node) const override {
            return parents_tostring(m_indices.at(node));
        }

        bool has_edge(node_descriptor source, node_descriptor dest) const {
            return g.has_edge(source, dest);
        }

        bool has_edge(int source, int dest) const override {
            return has_edge(g.node(source), g.node(dest));
        }

        bool has_edge(const std::string& source, const std::string& dest) const override {
            return has_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool has_path(node_descriptor source, node_descriptor dest) const {
            return g.has_path(source, dest);
        }
        
        bool has_path(int source_index, int dest_index) const override {
            return has_path(g.node(source_index), g.node(dest_index));
        }
        
        bool has_path(const std::string& source, const std::string& dest) const override {
            return has_path(m_indices.at(source), m_indices.at(dest));
        }

        void add_edge(node_descriptor source, node_descriptor dest) {
            g.add_edge(source, dest);
        }

        void add_edge(int source, int dest) override {
            add_edge(g.node(source), g.node(dest));
        }

        void add_edge(const std::string& source, const std::string& dest) override {
            add_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool can_add_edge(node_descriptor source, node_descriptor dest) const {
            if (num_parents(source) == 0 || num_children(dest) == 0 || !has_path(dest, source)) {
                return true;
            }

            return false;
        }

        bool can_add_edge(int source_index, int dest_index) const override {
            return can_add_edge(node(source_index), node(dest_index));
        }

        bool can_add_edge(const std::string& source, const std::string& dest) const override {
            return can_add_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool can_flip_edge(node_descriptor source, node_descriptor dest);

        bool can_flip_edge(int source_index, int dest_index) override {
            return can_flip_edge(node(source_index), node(dest_index));
        }

        bool can_flip_edge(const std::string& source, const std::string& dest) override {
            return can_flip_edge(m_indices.at(source), m_indices.at(dest));
        }

        void remove_edge(node_descriptor source, node_descriptor dest) {
            g.remove_edge(source, dest);
        }

        void remove_edge(int source, int dest) override {
            remove_edge(g.node(source), g.node(dest));
        }

        void remove_edge(const std::string& source, const std::string& dest) override {
            remove_edge(m_indices.at(source), m_indices.at(dest));
        }

        void check_blacklist(const ArcVector& arc_blacklist) const {
            for(auto& arc : arc_blacklist) {
                if (has_edge(arc.first, arc.second)) {
                    throw std::invalid_argument("Edge " + arc.first + " -> " + arc.second + " in blacklist,"
                                                " but it is present in the Bayesian Network.");
                }
            }
        }

        void force_whitelist(const ArcVector& arc_whitelist) {
            for(auto& arc : arc_whitelist) {
                if (!has_edge(arc.first, arc.second)) {
                    if (has_edge(arc.second, arc.first)) {
                        throw std::invalid_argument("Edge " + arc.first + " -> " + arc.second + " in whitelist,"
                                                    " but edge " + arc.second + " -> " + arc.first + " is present"
                                                    " in the Bayesian Network.");
                    } else {
                        add_edge(arc.first, arc.second);
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
            return cpd(m_indices.at(node));
        }

        VectorXd logpdf(const DataFrame& df) const override;
        double slogpdf(const DataFrame& df) const override;

        template<typename Derived_>
        friend std::ostream& operator<<(std::ostream &os, const BayesianNetwork<Derived_>& bn);

    protected:
        void check_fitted() const;
    private:
        DagType g;
        std::vector<std::string> m_nodes;
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;
        std::vector<CPD> m_cpds;
    };

    template<typename Derived_>
    std::ostream& operator<<(std::ostream &os, const BayesianNetwork<Derived_>& bn) {
        os << "Bayesian network: " << std::endl;
        for(auto [eit, eend] = bn.g.edges(); eit != eend; ++eit)
            os << bn.name(bn.g.source(*eit)) << " -> " << bn.name(bn.g.target(*eit)) << std::endl;
        return os;
    }

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size()) {
        if (nodes.empty()) {
            throw std::invalid_argument("Cannot define a BayesianNetwork without nodes");
        }
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }
    };

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const ArcVector& arcs) : g(0)
    {
        if (arcs.empty()) {
            throw std::invalid_argument("Cannot define a BayesianNetwork without nodes");
        }

        for (auto& arc : arcs) {
            if (m_indices.count(arc.first) == 0) {
                m_indices.insert(std::make_pair(arc.first, m_nodes.size()));
                m_nodes.push_back(arc.first);
            }

            if (m_indices.count(arc.second) == 0) {
                m_indices.insert(std::make_pair(arc.second, m_nodes.size()));
                m_nodes.push_back(arc.second);
            }
        }

        g = DagType(m_nodes.size());

        for(auto& arc : arcs) {
            g.add_edge(node(arc.first), node(arc.second));
        }

        g.check_acyclic();
    };

    template<typename Derived>
    BayesianNetwork<Derived>::BayesianNetwork(const std::vector<std::string>& nodes, 
                                              const ArcVector& edges) 
                                                 : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size())
    {
        if (nodes.empty()) {
            throw std::invalid_argument("Cannot define a BayesianNetwork without nodes");
        }
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }

        for(auto edge : edges) {
            if (m_indices.count(edge.first) == 0) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
            if (m_indices.count(edge.second) == 0) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            g.add_edge(node(edge.first), node(edge.second));
        }

        g.check_acyclic();
    };

    template<typename Derived>
    std::vector<std::string> BayesianNetwork<Derived>::parents(node_descriptor node) const {
        std::vector<std::string> parents;
        auto it_parents = g.get_parent_edges(node);

        for (auto it = it_parents.first; it != it_parents.second; ++it) {
            auto parent = g.source(*it);
            auto parent_index = g.index(parent);
            parents.push_back(m_nodes[parent_index]);
        }

        return parents;
    }

    template<typename Derived>
    std::vector<int> BayesianNetwork<Derived>::parent_indices(node_descriptor node) const {
        std::vector<int> parent_indices;
        auto it_parents = g.get_parent_edges(node);

        for (auto it = it_parents.first; it != it_parents.second; ++it) {
            parent_indices.push_back(g.index(g.source(*it)));
        }

        return parent_indices;
    }

    template<typename Derived>
    std::string BayesianNetwork<Derived>::parents_tostring(node_descriptor node) const {
        auto pa = parents(node);
        if (!pa.empty()) {
            std::string str = "[" + pa[0];
            for (auto it = pa.begin() + 1; it != pa.end(); ++it) {
                str += ", " + *it;
            }
            str += "]";
            return str;
        } else {
            return "[]";
        } 
    }

    template<typename Derived>
    bool BayesianNetwork<Derived>::can_flip_edge(node_descriptor source, node_descriptor dest) {
        if (num_parents(dest) == 0 || num_children(source) == 0) {
            return true;
        } else {
            remove_edge(source, dest);
            bool thereis_path = has_path(source, dest);
            add_edge(source, dest);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        }
    }

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
                                + "\nParents: " + parents_tostring(cpd.variable());

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
            m_cpds.reserve(m_nodes.size());

            for (auto& node : m_nodes) {
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
    VectorXd BayesianNetwork<Derived>::logpdf(const DataFrame& df) const {
        check_fitted();

        VectorXd accum = m_cpds[0].logpdf(df);
        for (auto it = ++m_cpds.begin(); it != m_cpds.end(); ++it) {
            accum += it->logpdf(df);
        }
        return accum;
    }

    template<typename Derived>
    double BayesianNetwork<Derived>::slogpdf(const DataFrame& df) const {
        check_fitted();
        
        double accum = m_cpds[0].slogpdf(df);
        for (auto it = ++m_cpds.begin(); it != m_cpds.end(); ++it) {
            accum += it->slogpdf(df);
        }
        return accum;
    }

    void requires_continuous_data(const DataFrame& df);
}

#endif //PGM_DATASET_BAYESIANNETWORK_HPP