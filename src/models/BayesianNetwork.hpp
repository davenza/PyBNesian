#ifndef PYBNESIAN_MODELS_BAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_BAYESIANNETWORK_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <graph/generic_graph.hpp>

using dataset::DataFrame;
using graph::Dag;
using util::ArcStringVector;

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

    // https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/
    template <typename Derived, typename Base>
    class clone_inherit : public Base {
    public:
        std::unique_ptr<Derived> clone() const {
            return std::unique_ptr<Derived>(static_cast<Derived *>(this->clone_impl()));
        }

    private:
        virtual clone_inherit * clone_impl() const = 0;
    };


    class BayesianNetworkBase : public clone_inherit<BayesianNetworkBase, BayesianNetworkBase> {
    public:
        virtual ~BayesianNetworkBase() = default;
        virtual int num_nodes() const = 0;
        virtual int num_arcs() const = 0;
        virtual std::vector<std::string> nodes() const = 0;
        virtual ArcStringVector arcs() const = 0;
        virtual const std::unordered_map<std::string, int>& indices() const = 0;
        virtual int index(const std::string& node) const = 0;
        virtual bool is_valid(int idx) const = 0;
        virtual bool contains_node(const std::string& name) const = 0;
        virtual size_t add_node(const std::string& node) = 0;
        virtual void remove_node(int node_index) = 0;
        virtual void remove_node(const std::string& node) = 0;
        virtual const std::string& name(int node_index) const = 0;
        virtual int num_parents(int node_index) const = 0;
        virtual int num_parents(const std::string& node) const = 0;
        virtual int num_children(int node_index) const = 0;
        virtual int num_children(const std::string& node) const = 0;
        virtual std::vector<std::string> parents(int node_index) const = 0;
        virtual std::vector<std::string> parents(const std::string& node) const = 0;
        virtual std::vector<int> parent_indices(int node_index) const = 0;
        virtual std::vector<int> parent_indices(const std::string& node) const = 0;
        virtual std::string parents_to_string(int node_index) const = 0;
        virtual std::string parents_to_string(const std::string& node) const = 0;
        virtual std::vector<std::string> children(int node_index) const = 0;
        virtual std::vector<std::string> children(const std::string& node) const = 0;
        virtual std::vector<int> children_indices(int node_index) const = 0;
        virtual std::vector<int> children_indices(const std::string& node) const = 0;
        virtual bool has_arc(int source, int dest) const = 0;
        virtual bool has_arc(const std::string& source, const std::string& dest) const = 0;
        virtual bool has_path(int source_index, int dest_index) const = 0;
        virtual bool has_path(const std::string& source, const std::string& dest) const = 0;
        virtual void add_arc(int source, int dest) = 0;
        virtual void add_arc(const std::string& source, const std::string& dest) = 0;
        virtual void remove_arc(int source, int dest) = 0;
        virtual void remove_arc(const std::string& source, const std::string& dest) = 0;
        virtual void flip_arc(int source, int dest) = 0;
        virtual void flip_arc(const std::string& source, const std::string& dest) = 0;
        virtual bool can_add_arc(int source_index, int dest_index) const = 0;
        virtual bool can_add_arc(const std::string& source, const std::string& dest) const = 0;
        virtual bool can_flip_arc(int source_index, int dest_index) = 0;
        virtual bool can_flip_arc(const std::string& source, const std::string& dest) = 0;
        void check_blacklist(const ArcStringVector& arc_blacklist) const {
            for(const auto& arc : arc_blacklist) {
                if (has_arc(arc.first, arc.second)) {
                    throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second + " in blacklist,"
                                                " but it is present in the Bayesian Network.");
                }
            }
        }
        
        void check_blacklist(const ArcSet& arc_blacklist) const {
            for(const auto& arc : arc_blacklist) {
                if (has_arc(arc.first, arc.second)) {
                    throw std::invalid_argument("Arc " + name(arc.first) + " -> " + name(arc.second) + " in blacklist,"
                                                " but it is present in the Bayesian Network.");
                }
            }
        }

        virtual void force_whitelist(const ArcStringVector& arc_whitelist) = 0;
        virtual void force_whitelist(const ArcSet& arc_whitelist) = 0;
        virtual bool fitted() const = 0;
        virtual void fit(const DataFrame& df) = 0;
        virtual VectorXd logl(const DataFrame& df) const = 0;
        virtual double slogl(const DataFrame& df) const = 0;
        virtual std::string ToString() const = 0;
        virtual BayesianNetworkType type() const = 0;
        virtual DataFrame sample(int n, unsigned int seed, bool ordered) const = 0;
        virtual void save(std::string name, bool include_cpd = false) const = 0;
        // virtual std::unique_ptr<BayesianNetworkBase> clone() const = 0;
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
        using CPD = typename BN_traits<Derived>::CPD;

        BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes), m_cpds() {}
        BayesianNetwork(const ArcStringVector& arcs) : g(arcs), m_cpds() {}
        BayesianNetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs) :
                                                 g(nodes, arcs), m_cpds() {}
        BayesianNetwork(const Dag& graph) : g(graph), m_cpds() {}
        BayesianNetwork(Dag&& graph) : g(std::move(graph)), m_cpds() {}

        int num_nodes() const override {
            return g.num_nodes();
        }

        int num_arcs() const override {
            return g.num_arcs();
        }

        std::vector<std::string> nodes() const override {
            return g.nodes();
        }

        ArcStringVector arcs() const override {
            return g.arcs();
        }

        const std::unordered_map<std::string, int>& indices() const override {
            return g.indices();
        }

        int index(const std::string& node) const override {
            return g.index(node);
        }

        bool is_valid(int idx) const override {
            return g.is_valid(idx);
        }

        bool contains_node(const std::string& name) const override {
            return g.contains_node(name);
        }

        size_t add_node(const std::string& node) override {
            size_t idx = g.add_node(node);

            if (idx >= m_cpds.size())
                m_cpds.resize(idx);
            return idx;
        }

        void remove_node(int node_index) override {
            g.remove_node(node_index);
            m_cpds[node_index] = CPD();
        }

        void remove_node(const std::string& node) override {
            auto idx = g.index(node);
            remove_node(idx);
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

        std::vector<std::string> children(int node_index) const override {
            return g.children(node_index);
        }

        std::vector<std::string> children(const std::string& node) const override {
            return g.children(node);
        }

        std::vector<int> children_indices(int node_index) const override {
            return g.children_indices(node_index);
        }

        std::vector<int> children_indices(const std::string& node) const override {
            return g.children_indices(node);
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

        void remove_arc(int source, int target) override {
            g.remove_arc(source, target);
        }

        void remove_arc(const std::string& source, const std::string& target) override {
            g.remove_arc(source, target);
        }

        void flip_arc(int source, int target) override {
            g.flip_arc(source, target);
        }

        void flip_arc(const std::string& source, const std::string& target) override {
            g.flip_arc(source, target);
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

        void force_whitelist(const ArcStringVector& arc_whitelist) override {
            for(const auto& arc : arc_whitelist) {
                if (!has_arc(arc.first, arc.second)) {
                    if (has_arc(arc.second, arc.first)) {
                        throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second + " in whitelist,"
                                                    " but arc " + arc.second + " -> " + arc.first + " is present"
                                                    " in the Bayesian Network.");
                    } else {
                        add_arc(arc.first, arc.second);
                    }
                }
            }

            g.topological_sort();
        }

        void force_whitelist(const ArcSet& arc_whitelist) override {
            for(const auto& arc : arc_whitelist) {
                if (!has_arc(arc.first, arc.second)) {
                    if (has_arc(arc.second, arc.first)) {
                        throw std::invalid_argument("Arc " + name(arc.first) + " -> " + name(arc.second) + " in whitelist,"
                                                    " but arc " + name(arc.second) + " -> " + name(arc.first) + " is present"
                                                    " in the Bayesian Network.");
                    } else {
                        add_arc(arc.first, arc.second);
                    }
                }
            }

            g.topological_sort();
        }

        bool fitted() const override;
        void compatible_cpd(const CPD& cpd) const;
        void add_cpds(const std::vector<CPD>& cpds);       
        bool must_construct_cpd(const CPD& node) const;
        void fit(const DataFrame& df) override;

        CPD create_cpd(const std::string& node) {
            auto pa = parents(node);
            return CPD(node, pa);
        }

        CPD& cpd(int index) {
            if (!m_cpds.empty() && is_valid(index))
                return m_cpds[index];
            else
                throw py::value_error("CPD of variable \"" + name(index) + "\" not added. Call add_cpds() or fit() to add the CPD.");
        }

        CPD& cpd(const std::string& node) {
            return cpd(g.index(node));
        }

        VectorXd logl(const DataFrame& df) const override;
        double slogl(const DataFrame& df) const override;
        DataFrame sample(int n,
                         unsigned int seed = std::random_device{}(),
                         bool ordered = false) const override;
        void save(std::string name, bool include_cpd = false) const override;

        py::tuple __getstate__() const;
        static Derived __setstate__(py::tuple& t);

        template<typename Derived_>
        friend std::ostream& operator<<(std::ostream &os, const BayesianNetwork<Derived_>& bn);
    protected:
        void check_fitted() const;
        size_t physical_num_nodes() const { return g.node_indices().size(); }
        int inner_index(const std::string& name) const {
            return index(name);
        }
    private:
        py::tuple __getstate_extra__() const {
            return py::make_tuple();
        }

        void __setstate_extra__(py::tuple&) const { }
        void __setstate_extra__(py::tuple&&) const { }

        Dag g;
        std::vector<CPD> m_cpds;
        // This is necessary because __getstate__() do not admit parameters.
        mutable bool m_include_cpd;
    };

    template<typename Derived>
    bool BayesianNetwork<Derived>::fitted() const {
        if (m_cpds.empty()) {
            return false;
        } else {
            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (is_valid(i) && !m_cpds[i].fitted()) {
                    return false;
                }
            }

            return true;
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

            m_cpds.reserve(physical_num_nodes());

            for (size_t i = 0; i < physical_num_nodes(); ++i) {
                if (is_valid(i)) {
                    auto cpd_idx = map_index.find(name(i));

                    if (cpd_idx != map_index.end()) {
                        auto cpd = *(cpd_idx->second);
                        m_cpds.push_back(cpd);
                    } else {
                        m_cpds.push_back(static_cast<Derived*>(this)->create_cpd(name(i)));
                    }
                } else {
                    m_cpds.push_back(CPD());
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
    bool BayesianNetwork<Derived>::must_construct_cpd(const CPD& cpd) const {
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
    void BayesianNetwork<Derived>::fit(const DataFrame& df) {
        if (m_cpds.empty()) {
            m_cpds.reserve(physical_num_nodes());

            for (size_t i = 0; i < physical_num_nodes(); ++i) {
                if (is_valid(i)) {
                    auto cpd = static_cast<Derived*>(this)->create_cpd(name(i));
                    m_cpds.push_back(cpd);
                    m_cpds.back().fit(df);
                } else {
                    m_cpds.push_back(CPD());
                }
            }
        } else {
            for (size_t i = 0; i < physical_num_nodes(); ++i) {
                if (is_valid(i)) {
                    if (static_cast<Derived*>(this)->must_construct_cpd(m_cpds[i])) {
                        m_cpds[i] = static_cast<Derived*>(this)->create_cpd(name(i));
                        m_cpds[i].fit(df);
                    } else if (!m_cpds[i].fitted()) {
                        m_cpds[i].fit(df);
                    }
                }
            }
        }
    }

    template<typename Derived>
    void BayesianNetwork<Derived>::check_fitted() const {
        if (m_cpds.empty()) {
            throw py::value_error("Model not fitted.");
        } else {
            bool all_fitted = true;
            std::string err;
            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (is_valid(i) && !m_cpds[i].fitted()) {
                    if (all_fitted) {
                        err += "Some CPDs are not fitted:\n";
                        all_fitted = false;
                    }
                    err += m_cpds[i].ToString() + "\n";
                }
            }

            if (!all_fitted)
                throw py::value_error(err);
        }
    }

    template<typename Derived>
    VectorXd BayesianNetwork<Derived>::logl(const DataFrame& df) const {
        check_fitted();

        size_t i = 0;
        for (size_t i = 0; i < m_cpds.size() && !is_valid(i); ++i);

        VectorXd accum = m_cpds[i].logl(df);

        for (++i; i < m_cpds.size(); ++i) {
            if (is_valid(i))
                accum += m_cpds[i].logl(df);
        }

        return accum;
    }

    template<typename Derived>
    double BayesianNetwork<Derived>::slogl(const DataFrame& df) const {
        check_fitted();
        
        double accum = 0;
        for (size_t i = 0; i < m_cpds.size(); ++i) {
            if (is_valid(i))
                accum += m_cpds[i].slogl(df);
        }
        return accum;
    }

    template<typename Derived>
    DataFrame BayesianNetwork<Derived>::sample(int n, unsigned int seed, bool ordered) const {
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

    template<typename Derived>
    void BayesianNetwork<Derived>::save(std::string name, bool include_cpd) const {
        m_include_cpd = include_cpd;
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(static_cast<const Derived*>(this)), file, 2);
        file.attr("close")();
    }

    template<typename Derived>
    py::tuple BayesianNetwork<Derived>::__getstate__() const {
        auto g_tuple = g.__getstate__();

        auto extra_info = static_cast<const Derived&>(*this).__getstate_extra__();

        if (m_include_cpd && !m_cpds.empty()) {
            std::vector<py::tuple> cpds;
            cpds.reserve(g.num_nodes());

            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (g.is_valid(i))
                    cpds.push_back(m_cpds[i].__getstate__());
            }

            return py::make_tuple(g_tuple, true, cpds, extra_info);
        } {
            return py::make_tuple(g_tuple, false, py::make_tuple(), extra_info);
        }
    }

    template<typename Derived>
    Derived BayesianNetwork<Derived>::__setstate__(py::tuple& t) {
        if (t.size() != 4)
            throw std::runtime_error("Not valid BayesianNetwork.");
        
        auto bn = Derived(Dag::__setstate__(t[0].cast<py::tuple>()));

        bn.__setstate_extra__(t[3].cast<py::tuple>());

        if (t[1].cast<bool>()) {
            auto py_cpds = t[2].cast<std::vector<py::tuple>>();
            std::vector<CPD> cpds;

            for (auto& py_cpd : py_cpds) {
                cpds.push_back(CPD::__setstate__(py_cpd));
            }

            bn.add_cpds(cpds);
        }

        return bn;
    }

    py::object load_model(const std::string& name);

    void requires_continuous_data(const DataFrame& df);
    void requires_discrete_data(const DataFrame& df);
}

#endif //PYBNESIAN_MODELS_BAYESIANNETWORK_HPP