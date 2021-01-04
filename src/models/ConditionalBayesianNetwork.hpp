#ifndef PYBNESIAN_MODELS_CONDITIONALBAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_CONDITIONALBAYESIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <util/vector.hpp>


namespace models {

    class ConditionalBayesianNetworkBase : public BayesianNetworkBase {
    public:
        virtual ~ConditionalBayesianNetworkBase() = default;
        virtual int num_interface_nodes() const = 0;
        virtual int num_total_nodes() const = 0;
        virtual std::vector<std::string> interface_nodes() const = 0;
        virtual std::vector<std::string> all_nodes() const = 0;
        virtual const std::unordered_map<std::string, int>& interface_indices() const = 0;
        virtual bool contains_interface_node(const std::string& name) const = 0;
        virtual bool contains_all_node(const std::string& name) const = 0;
        virtual size_t add_interface_node(const std::string& node) = 0;
        virtual void remove_interface_node(int node_index) = 0;
        virtual void remove_interface_node(const std::string& node) = 0;
        virtual bool is_interface(const std::string& name) const = 0;
        virtual bool is_interface(int index) const = 0;
        using BayesianNetworkBase::sample;
        virtual DataFrame sample(const DataFrame& evidence, unsigned int seed, bool concat_evidence, bool ordered) const = 0;
        // std::unique_ptr<ConditionalBayesianNetworkBase> clone() const = 0;
    private:
    };


    template<typename Derived>
    class ConditionalBayesianNetwork : public ConditionalBayesianNetworkBase {
    public:
        using CPD = typename BN_traits<Derived>::CPD;

        ConditionalBayesianNetwork(const std::vector<std::string>& nodes, 
                                   const std::vector<std::string>& interface_nodes) : g(),
                                                                                      m_cpds(),
                                                                                      m_cpds_indices(),
                                                                                      m_nodes(nodes),
                                                                                      m_indices(),
                                                                                      m_interface_nodes(interface_nodes),
                                                                                      m_interface_indices()
        {
            if (nodes.empty()) {
                throw std::invalid_argument("Nodes can not be empty.");
            }

            std::vector<std::string> all_nodes (nodes);
            all_nodes.reserve(nodes.size() + interface_nodes.size());
            all_nodes.insert(all_nodes.end(), interface_nodes.begin(), interface_nodes.end());

            g = Dag(all_nodes);

            for (int i = 0; i < m_nodes.size(); ++i) {
                m_cpds_indices.insert({m_nodes[i], i});
                m_indices.insert({m_nodes[i], g.index(m_nodes[i])});
            }

            for (const auto& n : m_interface_nodes) {
                m_interface_indices.insert({n, g.index(n)});
            }
        }

        ConditionalBayesianNetwork(const std::vector<std::string>& nodes, 
                                   const std::vector<std::string>& interface_nodes,
                                   const ArcStringVector& arcs) :
                                            ConditionalBayesianNetwork<Derived>(nodes, interface_nodes) 
        {    
            for (const auto& arc : arcs) {
                if (is_interface(arc.second)) {
                    throw std::invalid_argument("Interface can not have parents. "
                                            "Error in arc: (" + arc.first + ", " + arc.second + ")");
                }

                g.add_arc(arc.first, arc.second);
            }
        }

        ConditionalBayesianNetwork(const std::vector<std::string>& nodes, 
                                   const std::vector<std::string>& interface_nodes,
                                   const Dag& graph) : g(graph),
                                                       m_cpds(),
                                                       m_cpds_indices(),
                                                       m_nodes(nodes),
                                                       m_indices(),
                                                       m_interface_nodes(interface_nodes),
                                                       m_interface_indices()
        {
            
            for (int i = 0; i < m_nodes.size(); ++i) {
                m_cpds_indices.insert({m_nodes[i], i});
                m_indices.insert({m_nodes[i], g.index(m_nodes[i])});
            }

            for (const auto& n : m_interface_nodes) {
                m_interface_indices.insert({n, g.index(n)});
            }

            if (g.num_nodes() != (m_indices.size() + m_interface_indices.size())) {
                throw std::invalid_argument("The number of nodes in the graph and "
                                "the number of names in lists of nodes/interface nodes is different.");
            }
        }

        ConditionalBayesianNetwork(const std::vector<std::string>& nodes, 
                                   const std::vector<std::string>& interface_nodes,
                                   Dag&& graph) : g(std::move(graph)),
                                                  m_cpds(),
                                                  m_cpds_indices(),
                                                  m_nodes(nodes),
                                                  m_indices(),
                                                  m_interface_nodes(interface_nodes),
                                                  m_interface_indices() 
        {
            for (int i = 0; i < m_nodes.size(); ++i) {
                m_cpds_indices.insert({m_nodes[i], i});
                m_indices.insert({m_nodes[i], g.index(m_nodes[i])});
            }

            for (const auto& n : m_interface_nodes) {
                m_interface_indices.insert({n, g.index(n)});
            }

            if (g.num_nodes() != (m_indices.size() + m_interface_indices.size())) {
                throw std::invalid_argument("The number of nodes in the graph and "
                                "the number of names in lists of nodes/interface nodes is different.");
            }
        }

        int num_nodes() const override {
            return m_nodes.size();
        }

        int num_interface_nodes() const override {
            return m_interface_nodes.size();
        }

        int num_total_nodes() const override {
            return g.num_nodes();
        }

        int num_arcs() const override {
            return g.num_arcs();
        }

        std::vector<std::string> nodes() const override {
            return m_nodes;
        }

        std::vector<std::string> interface_nodes() const override {
            return m_interface_nodes;
        }

        std::vector<std::string> all_nodes() const override {
            return g.nodes();
        }
    
        ArcStringVector arcs() const override {
            return g.arcs();
        }

        const std::unordered_map<std::string, int>& indices() const override {
            return m_indices;
        }

        const std::unordered_map<std::string, int>& interface_indices() const override {
            return m_interface_indices;
        }

        int index(const std::string& node) const override {
            return g.index(node);
        }

        bool is_valid(int idx) const override {
            return g.is_valid(idx);
        }

        bool contains_node(const std::string& name) const override {
            return m_indices.count(name) > 0;
        }

        bool contains_interface_node(const std::string& name) const override {
            return m_interface_indices.count(name) > 0;
        };

        bool contains_all_node(const std::string& name) const override {
            return g.contains_node(name);
        }

        size_t add_node(const std::string& node) override {
            auto new_index = g.add_node(node);
            m_nodes.push_back(node);
            m_cpds_indices.insert({node, m_nodes.size()-1});
            m_indices.insert({node, new_index});

            if (m_nodes.size() >= m_cpds.size())
                m_cpds.resize(m_nodes.size());
            return new_index;
        }

        size_t add_interface_node(const std::string& node) override {
            auto new_index = g.add_node(node);
            m_interface_indices.insert({node, new_index});
            return new_index;
        }

        void remove_node(int node_index) override {
            remove_node(name(node_index));
        }

        void remove_interface_node(int node_index) override {
            remove_interface_node(name(node_index));
        }

        void remove_node(const std::string& node) override {
            if (!contains_node(node)) {
                throw std::invalid_argument("ConditionalBayesianNetwork does not contain node " + node);
            }

            g.remove_node(node);

            auto inner_index = m_cpds_indices.at(node);
            util::swap_remove(m_nodes, inner_index);
            m_indices.erase(node);
            m_cpds_indices.erase(node);

            // Update cpds indices if swap remove was performed.
            if (inner_index < m_nodes.size()) {
                m_cpds_indices[m_nodes[inner_index]] = inner_index;
            }

            if (!m_cpds.empty()) {
                util::swap_remove(m_cpds, inner_index);
            }
        }

        void remove_interface_node(const std::string& node) override {
            if (!contains_interface_node(node)) {
                throw std::invalid_argument("ConditionalBayesianNetwork does not contain interface node " + node);
            }

            g.remove_node(node);

            util::swap_remove_v(m_interface_nodes, node);
            m_interface_indices.erase(node);
        }
        
        bool is_interface(const std::string& name) const override {
            return m_interface_indices.count(name) > 0;
        }

        bool is_interface(int index) const override {
            return is_interface(name(index));
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
            if (is_interface(target))
                throw std::invalid_argument("Interface node " + std::to_string(target) + " cannot have parents.");
            g.add_arc(source, target);
        }

        void add_arc(const std::string& source, const std::string& target) override {
            if (is_interface(target))
                throw std::invalid_argument("Interface node " + target + " cannot have parents.");
            g.add_arc(source, target);
        }

        void remove_arc(int source, int target_index) override {
            g.remove_arc(source, target_index);
        }

        void remove_arc(const std::string& source, const std::string& target) override {
            g.remove_arc(source, target);
        }

        void flip_arc(int source, int target) override {
            if (is_interface(source))
                throw std::invalid_argument("Interface node " + std::to_string(source) + " cannot have parents.");
            
            g.flip_arc(source, target);
        }

        void flip_arc(const std::string& source, const std::string& target) override {
            if (is_interface(source))
                throw std::invalid_argument("Interface node " + source + " cannot have parents.");
            
            g.flip_arc(source, target);
        }

        bool can_add_arc(int source_index, int target_index) const override {
            return !is_interface(target_index) && g.can_add_arc(source_index, target_index);
        }

        bool can_add_arc(const std::string& source, const std::string& target) const override {
            return !is_interface(target) && g.can_add_arc(source, target);
        }

        bool can_flip_arc(int source_index, int target_index) override {
            return !is_interface(source_index) && g.can_flip_arc(source_index, target_index);
        }

        bool can_flip_arc(const std::string& source, const std::string& target) override {
            return !is_interface(source) && g.can_flip_arc(source, target);
        }

        void force_whitelist(const ArcStringVector& arc_whitelist) override {
            for(const auto& arc : arc_whitelist) {
                if (is_interface(arc.second))
                    throw std::invalid_argument("Interface node " + arc.second + " cannot have parents.");


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
                if (is_interface(arc.second))
                    throw std::invalid_argument("Interface node " + name(arc.second) + " cannot have parents.");

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
            cpd(name(index));
        }

        CPD& cpd(const std::string& node) {
            if (is_interface(node)) {
                throw py::value_error("Node \"" + node + "\" is interface, so it cannot have a CPD.");
            }

            if (!m_cpds.empty()) {
                return m_cpds[m_cpds_indices.at(node)];
            } else {
                throw py::value_error("CPD of variable \"" + node + "\" not added. Call add_cpds() or fit() to add the CPD.");
            }
        }

        VectorXd logl(const DataFrame& df) const override;
        double slogl(const DataFrame& df) const override;
        DataFrame sample(int n, unsigned int seed, bool ordered) const override;
        DataFrame sample(const DataFrame& evidence,
                         unsigned int seed = std::random_device{}(),
                         bool concat_evidence = true,
                         bool ordered = false) const override;
        void save(std::string name, bool include_cpd = false) const override;

        py::tuple __getstate__() const;
        static Derived __setstate__(py::tuple& t);
    protected:
        void check_fitted() const;
        int inner_index(const std::string& name) const {
            return m_cpds_indices.at(name);
        }
    private:
        py::tuple __getstate_extra__() const {
            return py::make_tuple();
        }

        void __setstate_extra__(py::tuple&) const { }
        void __setstate_extra__(py::tuple&&) const { }

        Dag g;
        std::vector<CPD> m_cpds;
        std::unordered_map<std::string, int> m_cpds_indices;
        std::vector<std::string> m_nodes;
        std::unordered_map<std::string, int> m_indices;
        std::vector<std::string> m_interface_nodes;
        std::unordered_map<std::string, int> m_interface_indices;
        // This is necessary because __getstate__() do not admit parameters.
        mutable bool m_include_cpd;
    };

    template<typename Derived>
    bool ConditionalBayesianNetwork<Derived>::fitted() const {
        if (m_cpds.empty()) {
            return false;
        } else {
            for (const auto& cpd : m_cpds) {
                if (!cpd.fitted()) {
                    return false;
                }
            }

            return true;
        }    
    }

    template<typename Derived>
    void ConditionalBayesianNetwork<Derived>::compatible_cpd(const CPD& cpd) const {
        if (!contains_node(cpd.variable())) {
            throw std::invalid_argument("CPD defined on variable which is not present in the model:\n" + cpd.ToString());
        }

        auto& evidence = cpd.evidence();

        for (auto& ev : evidence) {
            if (!contains_all_nodes(ev)) {
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
    void ConditionalBayesianNetwork<Derived>::add_cpds(const std::vector<CPD>& cpds) {
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

            for (size_t i = 0; i < num_nodes(); ++i) {
                auto cpd_idx = map_index.find(m_nodes[i]);

                if (cpd_idx != map_index.end()) {
                    m_cpds.push_back(*(cpd_idx->second));
                } else {
                    m_cpds.push_back(static_cast<Derived*>(this)->create_cpd(m_nodes[i]));
                }
            }
        } else {
            for(auto& cpd : cpds) {
                auto idx = m_cpds_indices.at(cpd.variable());
                m_cpds[idx] = cpd;
            }
        }
    }

    template<typename Derived>
    bool ConditionalBayesianNetwork<Derived>::must_construct_cpd(const CPD& cpd) const {
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
    void ConditionalBayesianNetwork<Derived>::fit(const DataFrame& df) {
        if (m_cpds.empty()) {
            m_cpds.reserve(num_nodes());

            for (size_t i = 0; i < num_nodes(); ++i) {
                auto cpd = static_cast<Derived*>(this)->create_cpd(m_nodes[i]);
                m_cpds.push_back(cpd);
                m_cpds.back().fit(df);
            }
        } else {
            for (size_t i = 0; i < num_nodes(); ++i) {
                if (static_cast<Derived*>(this)->must_construct_cpd(m_cpds[i])) {
                    m_cpds[i] = static_cast<Derived*>(this)->create_cpd(m_nodes[i]);
                    m_cpds[i].fit(df);
                } else if (!m_cpds[i].fitted()) {
                    m_cpds[i].fit(df);
                }
            }
        }
    }

    template<typename Derived>
    void ConditionalBayesianNetwork<Derived>::check_fitted() const {
        if (m_cpds.empty()) {
            throw py::value_error("Model not fitted.");
        } else {
            bool all_fitted = true;
            std::string err;
            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (!m_cpds[i].fitted()) {
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
    VectorXd ConditionalBayesianNetwork<Derived>::logl(const DataFrame& df) const {
        check_fitted();

        VectorXd accum = m_cpds[0].logl(df);

        for (size_t i = 1; i < m_cpds.size(); ++i) {
            accum += m_cpds[i].logl(df);
        }

        return accum;
    }


    template<typename Derived>
    double ConditionalBayesianNetwork<Derived>::slogl(const DataFrame& df) const {
        check_fitted();
        
        double accum = 0;
        for (size_t i = 0; i < m_cpds.size(); ++i) {
            accum += m_cpds[i].slogl(df);
        }

        return accum;
    }

    template<typename Derived>
    DataFrame ConditionalBayesianNetwork<Derived>::sample(int, unsigned int, bool) const {
        throw std::runtime_error("Can not sample from ConditionalBayesianNetwork "
                                 "if evidence is not provided for the interface nodes.");
    }

    template<typename Derived>
    DataFrame ConditionalBayesianNetwork<Derived>::sample(const DataFrame& evidence,
                                                          unsigned int seed,
                                                          bool concat_evidence,
                                                          bool ordered) const {
        check_fitted();
        evidence.has_columns(m_interface_nodes);
        
        DataFrame parents(evidence);

        auto top_sort = g.topological_sort();

        for (auto i = 0; i < top_sort.size(); ++i) {
            if (!is_interface(top_sort[i])) {
                auto array = m_cpds[m_cpds_indices.at(top_sort[i])].sample(evidence->num_rows(), parents, seed);

                auto res = parents->AddColumn(evidence->num_columns() + i, top_sort[i], array);
                parents = DataFrame(std::move(res).ValueOrDie());
            }
        }

        std::vector<Field_ptr> fields;
        std::vector<Array_ptr> columns;

        auto schema = parents->schema();
        if (ordered) {
            for (const auto& name : m_nodes) {
                fields.push_back(schema->GetFieldByName(name));
                columns.push_back(parents.col(name));
            }
        } else {
            for (auto i = evidence->num_columns(); i < parents->num_columns(); ++i) {
                fields.push_back(schema->field(i));
                columns.push_back(parents.col(i));
            }
        }

        if (concat_evidence) {
            auto evidence_schema = evidence->schema();
            for (auto i = 0; i < evidence->num_columns(); ++i) {
                fields.push_back(evidence_schema->field(i));
                columns.push_back(evidence.col(i));
            }
        }

        auto new_schema = std::make_shared<arrow::Schema>(fields);
        auto new_rb = arrow::RecordBatch::Make(new_schema, evidence->num_rows(), columns);
        return DataFrame(new_rb);
    }

    template<typename Derived>
    void ConditionalBayesianNetwork<Derived>::save(std::string name, bool include_cpd) const {
        m_include_cpd = include_cpd;
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(static_cast<const Derived*>(this)), file, 2);
        file.attr("close")();
    }

    template<typename Derived>
    py::tuple ConditionalBayesianNetwork<Derived>::__getstate__() const {
        auto g_tuple = g.__getstate__();

        auto extra_info = static_cast<const Derived&>(*this).__getstate_extra__();

        if (m_include_cpd && !m_cpds.empty()) {
            std::vector<py::tuple> cpds;
            cpds.reserve(num_nodes());

            for (size_t i = 0; i < m_cpds.size(); ++i) {
                cpds.push_back(m_cpds[i].__getstate__());
            }

            return py::make_tuple(g_tuple, true, cpds, m_nodes, m_interface_nodes, extra_info);
        } {
            return py::make_tuple(g_tuple, false, py::make_tuple(), m_nodes, m_interface_nodes, extra_info);
        }
    }

    template<typename Derived>
    Derived ConditionalBayesianNetwork<Derived>::__setstate__(py::tuple& t) {
        if (t.size() != 6)
            throw std::runtime_error("Not valid ConditionalBayesianNetwork.");
        
        auto nodes = t[3].cast<std::vector<std::string>>();
        auto interface_nodes = t[4].cast<std::vector<std::string>>();

        auto bn = Derived(nodes, interface_nodes, Dag::__setstate__(t[0].cast<py::tuple>()));

        bn.__setstate_extra__(t[5].cast<py::tuple>());

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

}


#endif //PYBNESIAN_MODELS_CONDITIONALBAYESIANNETWORK_HPP