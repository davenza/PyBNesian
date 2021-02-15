#ifndef PYBNESIAN_MODELS_BAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_BAYESIANNETWORK_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <graph/generic_graph.hpp>
#include <util/virtual_clone.hpp>

using dataset::DataFrame;
using graph::Dag, graph::ConditionalDag;
using util::ArcStringVector, util::FactorStringTypeVector;

using util::abstract_class, util::clone_inherit;

using Field_ptr = std::shared_ptr<arrow::Field>;
using Array_ptr = std::shared_ptr<arrow::Array>;

namespace models {

    class BayesianNetworkType
    {
    public:
        enum Value : uint8_t
        {
            Gaussian,
            Semiparametric,
            Discrete
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
                case Value::Gaussian:
                    return "Gaussian";
                case Value::Semiparametric:
                    return "Semiparametric";
                case Value::Discrete:
                    return "Discrete";
                default:
                    throw std::invalid_argument("Unreachable code in BayesianNetworkType.");
            }
        }
    private:
        Value value;
    };

    class ConditionalBayesianNetworkBase;

    class BayesianNetworkBase : public clone_inherit<abstract_class<BayesianNetworkBase>> {
    public:
        virtual ~BayesianNetworkBase() = default;
        virtual int num_nodes() const = 0;
        virtual int num_arcs() const = 0;
        virtual const std::vector<std::string>& nodes() const = 0;
        virtual ArcStringVector arcs() const = 0;
        virtual const std::unordered_map<std::string, int>& indices() const = 0;
        virtual int index(const std::string& node) const = 0;
        virtual int collapsed_index(const std::string& node) const = 0;
        virtual int index_from_collapsed(int collapsed_index) const = 0;
        virtual int collapsed_from_index(int index) const = 0;
        virtual const std::unordered_map<std::string, int>& collapsed_indices() const = 0;
        virtual bool is_valid(int idx) const = 0;
        virtual bool contains_node(const std::string& name) const = 0;
        virtual int add_node(const std::string& node) = 0;
        virtual void remove_node(int node_index) = 0;
        virtual void remove_node(const std::string& node) = 0;
        virtual const std::string& name(int node_index) const = 0;
        virtual const std::string& collapsed_name(int collapsed_index) const = 0;
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
        virtual bool can_flip_arc(int source_index, int dest_index) const = 0;
        virtual bool can_flip_arc(const std::string& source, const std::string& dest) const = 0;
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
        virtual std::unique_ptr<ConditionalBayesianNetworkBase> 
        conditional_bn(const std::vector<std::string>& nodes,
                       const std::vector<std::string>& interface_nodes) const = 0;
        virtual std::unique_ptr<ConditionalBayesianNetworkBase> conditional_bn() const = 0;
        virtual std::unique_ptr<BayesianNetworkBase> unconditional_bn() const = 0;
        virtual void save(std::string name, bool include_cpd = false) const = 0;
    };

    class ConditionalBayesianNetworkBase : public clone_inherit<abstract_class<ConditionalBayesianNetworkBase>, 
                                                                BayesianNetworkBase>  {
    public:
        virtual ~ConditionalBayesianNetworkBase() = default;
        virtual int num_interface_nodes() const = 0;
        virtual int num_total_nodes() const = 0;
        virtual const std::vector<std::string>& interface_nodes() const = 0;
        virtual const std::vector<std::string>& all_nodes() const = 0;
        virtual int interface_collapsed_index(const std::string& name) const = 0;
        virtual int joint_collapsed_index(const std::string& name) const = 0;
        virtual const std::unordered_map<std::string, int>& interface_collapsed_indices() const = 0;
        virtual const std::unordered_map<std::string, int>& joint_collapsed_indices() const = 0;
        virtual int index_from_interface_collapsed(int interface_collapsed_index) const = 0;
        virtual int index_from_joint_collapsed(int joint_collapsed_index) const = 0;
        virtual int interface_collapsed_from_index(int index) const = 0;
        virtual int joint_collapsed_from_index(int index) const = 0;
        virtual const std::string& interface_collapsed_name(int interface_collapsed_index) const = 0;
        virtual const std::string& joint_collapsed_name(int joint_collapsed_index) const = 0;
        virtual bool contains_interface_node(const std::string& name) const = 0;
        virtual bool contains_total_node(const std::string& name) const = 0;
        virtual int add_interface_node(const std::string& node) = 0;
        virtual void remove_interface_node(int node_index) = 0;
        virtual void remove_interface_node(const std::string& node) = 0;
        virtual bool is_interface(int index) const = 0;
        virtual bool is_interface(const std::string& name) const = 0;
        virtual void set_interface(int index) = 0;
        virtual void set_interface(const std::string& name) = 0;
        virtual void set_node(int index) = 0;
        virtual void set_node(const std::string& name) = 0;
        using BayesianNetworkBase::sample;
        virtual DataFrame sample(const DataFrame& evidence, unsigned int seed, bool concat_evidence, bool ordered) const = 0;
    };

    class SemiparametricBNBase {
    public:
        virtual ~SemiparametricBNBase() = default;
        virtual FactorType node_type(int node_index) const = 0;
        virtual FactorType node_type(const std::string& node) const = 0;
        virtual std::unordered_map<std::string, FactorType> node_types() const = 0;
        virtual void set_node_type(int node_index, FactorType new_type) = 0;
        virtual void set_node_type(const std::string& node, FactorType new_type) = 0;
        virtual void force_type_whitelist(const FactorStringTypeVector& type_whitelist) = 0;
    };

    template<BayesianNetworkType::Value Type>
    class BayesianNetwork;
    template<BayesianNetworkType::Value Type>
    class ConditionalBayesianNetwork;

    using GaussianNetwork = BayesianNetwork<BayesianNetworkType::Gaussian>;
    using SemiparametricBN = BayesianNetwork<BayesianNetworkType::Semiparametric>;
    using DiscreteBN = BayesianNetwork<BayesianNetworkType::Discrete>;

    using ConditionalGaussianNetwork = ConditionalBayesianNetwork<BayesianNetworkType::Gaussian>;
    using ConditionalSemiparametricBN = ConditionalBayesianNetwork<BayesianNetworkType::Semiparametric>;
    using ConditionalDiscreteBN = ConditionalBayesianNetwork<BayesianNetworkType::Discrete>;

    template<typename Model>
    struct BN_traits {};

    template<typename G, typename _ = void>
    struct is_unconditional_bn : public std::false_type {};

    template<typename G>
    struct is_unconditional_bn<G,
                               std::void_t<
                                    util::GenericInstantation<BayesianNetworkType::Value>::
                                    enable_if_template_instantation_t<BayesianNetwork, G>
                               >
    > : public std::true_type {};

    template<typename G>
    inline constexpr auto is_unconditional_bn_v = is_unconditional_bn<G>::value;

    template<typename G, typename R = void>
    using enable_if_unconditional_bn_t = std::enable_if_t<is_unconditional_bn_v<G>, R>;

    template<typename G, typename _ = void>
    struct is_conditional_bn : public std::false_type {};

    template<typename G>
    struct is_conditional_bn<G,
                                std::void_t<
                                    util::GenericInstantation<BayesianNetworkType::Value>::
                                    enable_if_template_instantation_t<ConditionalBayesianNetwork, G>
                                >
    > : public std::true_type {};

    template<typename G>
    inline constexpr auto is_conditional_bn_v = is_conditional_bn<G>::value;

    template<typename G, typename R = void>
    using enable_if_conditional_bn_t = std::enable_if_t<is_conditional_bn_v<G>, R>;

    template<typename Derived>
    class ConditionalBayesianNetworkImpl;


    template<typename Derived>
    class BayesianNetworkImpl : public BN_traits<Derived>::BaseClass {
    public:
        using CPD = typename BN_traits<Derived>::CPD;
        using DagClass = typename BN_traits<Derived>::DagClass;
        template<BayesianNetworkType::Value Type>
        using BNClass = typename BN_traits<Derived>::template BNClass<Type>;

        friend class ConditionalBayesianNetworkImpl<Derived>;

        BayesianNetworkImpl() = default;

        // /////////////////////////////////////
        // Unconditional BN constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        BayesianNetworkImpl(const std::vector<std::string>& nodes) : g(nodes), m_cpds() {}
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        BayesianNetworkImpl(const ArcStringVector& arcs) : g(arcs), m_cpds() {}
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        BayesianNetworkImpl(const std::vector<std::string>& nodes, const ArcStringVector& arcs) 
                                                        : g(nodes, arcs), m_cpds() {}

        // /////////////////////////////////////
        // Conditional BN constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        BayesianNetworkImpl(const std::vector<std::string>& nodes,
                            const std::vector<std::string>& interface_nodes) : g(nodes, interface_nodes), 
                                                                               m_cpds() {}

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        BayesianNetworkImpl(const std::vector<std::string>& nodes,
                            const std::vector<std::string>& interface_nodes,
                            const ArcStringVector& arcs) : g(nodes, interface_nodes, arcs), 
                                                           m_cpds() {}



        BayesianNetworkImpl(const DagClass& graph) : g(graph), m_cpds() {}
        BayesianNetworkImpl(DagClass&& graph) : g(std::move(graph)), m_cpds() {}

        const DagClass& graph() const {
            return g;
        }

        int num_nodes() const override {
            return g.num_nodes();
        }

        int num_raw_nodes() const { return g.num_raw_nodes(); }

        int num_arcs() const override {
            return g.num_arcs();
        }

        const std::vector<std::string>& nodes() const override {
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

        int check_index(int idx) const {
            return g.check_index(idx);
        }

        int check_index(const std::string& name) const {
            return g.check_index(name);
        }

        int collapsed_index(const std::string& node) const override {
            return g.collapsed_index(node);
        }

        int index_from_collapsed(int collapsed_index) const override {
            return g.index_from_collapsed(collapsed_index);
        }

        int collapsed_from_index(int index) const override {
            return g.collapsed_from_index(index);
        }

        const std::unordered_map<std::string, int>& collapsed_indices() const override {
            return g.collapsed_indices();
        }

        bool is_valid(int idx) const override {
            return g.is_valid(idx);
        }

        bool contains_node(const std::string& name) const override {
            return g.contains_node(name);
        }

        int add_node(const std::string& node) override {
            int idx = g.add_node(node);

            if (!m_cpds.empty() && static_cast<size_t>(idx) >= m_cpds.size())
                m_cpds.resize(idx + 1);
            return idx;
        }

        void remove_node(int node_index) override {
            g.remove_node(node_index);
            if (!m_cpds.empty()) {
                m_cpds[node_index] = CPD();
            }
        }

        void remove_node(const std::string& node) override {
            auto idx = g.index(node);
            remove_node(idx);
        }

        const std::string& name(int node_index) const override {
            return g.name(node_index);
        }

        const std::string& collapsed_name(int collapsed_index) const override {
            return g.collapsed_name(collapsed_index);
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

        bool can_flip_arc(int source_index, int target_index) const override {
            return g.can_flip_arc(source_index, target_index);
        }

        bool can_flip_arc(const std::string& source, const std::string& target) const override {
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

        bool can_have_cpd(int index) const {
            return is_valid(index);
        }

        bool can_have_cpd(const std::string& name) const {
            return is_valid(index(name));
        }

        bool fitted() const override;
        void compatible_cpd(const CPD& cpd) const;
        void add_cpds(const std::vector<CPD>& cpds);       
        bool must_construct_cpd(const CPD& node) const;
        void fit(const DataFrame& df) override;

        CPD create_cpd(const std::string& node) const {
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

        const CPD& cpd(int index) const {
            if (!m_cpds.empty() && is_valid(index))
                return m_cpds[index];
            else
                throw py::value_error("CPD of variable \"" + name(index) + "\" not added. Call add_cpds() or fit() to add the CPD.");
        }

        const CPD& cpd(const std::string& node) const {
            return cpd(g.index(node));
        }

        VectorXd logl(const DataFrame& df) const override;
        double slogl(const DataFrame& df) const override;

        BayesianNetworkType type() const override {
            return Derived::TYPE;
        }

        using BN_traits<Derived>::BaseClass::sample;
        DataFrame sample(int n,
                         unsigned int seed = std::random_device{}(),
                         bool ordered = false) const override;


        std::unique_ptr<ConditionalBayesianNetworkBase> 
        conditional_bn(const std::vector<std::string>& nodes,
                       const std::vector<std::string>& interface_nodes) const override;
        std::unique_ptr<ConditionalBayesianNetworkBase> conditional_bn() const override;
        std::unique_ptr<BayesianNetworkBase> unconditional_bn() const override;

        void save(std::string name, bool include_cpd = false) const override;
        py::tuple __getstate__() const;
        static Derived __setstate__(py::tuple& t);
    protected:
        void check_fitted() const;
    private:
        py::tuple __getstate_extra__() const {
            return py::make_tuple();
        }

        void __setstate_extra__(py::tuple&) const { }
        void __setstate_extra__(py::tuple&&) const { }

        DagClass g;
        std::vector<CPD> m_cpds;
        // This is necessary because __getstate__() do not admit parameters.
        mutable bool m_include_cpd;
    };

    template<typename Derived>
    class ConditionalBayesianNetworkImpl : public BayesianNetworkImpl<Derived> {
    public:
        using CPD = typename BN_traits<Derived>::CPD;
        using BayesianNetworkImpl<Derived>::BayesianNetworkImpl;

        int num_interface_nodes() const override {
            return this->g.num_interface_nodes();
        }

        int num_total_nodes() const override {
            return this->g.num_total_nodes();
        }

        const std::vector<std::string>& interface_nodes() const override {
            return this->g.interface_nodes();
        }
        
        const std::vector<std::string>& all_nodes() const override {
            return this->g.all_nodes();
        }

        int interface_collapsed_index(const std::string& name) const override {
            return this->g.interface_collapsed_index(name);
        }

        int joint_collapsed_index(const std::string& name) const override {
            return this->g.joint_collapsed_index(name);
        }

        const std::unordered_map<std::string, int>& interface_collapsed_indices() const override {
            return this->g.interface_collapsed_indices();
        }

        const std::unordered_map<std::string, int>& joint_collapsed_indices() const override {
            return this->g.joint_collapsed_indices();
        }

        int index_from_interface_collapsed(int interface_collapsed_index) const override {
            return this->g.index_from_interface_collapsed(interface_collapsed_index);
        }

        int index_from_joint_collapsed(int joint_collapsed_index) const override {
            return this->g.index_from_joint_collapsed(joint_collapsed_index);
        }

        int interface_collapsed_from_index(int index) const override {
            return this->g.interface_collapsed_from_index(index);
        }

        int joint_collapsed_from_index(int index) const override {
            return this->g.joint_collapsed_from_index(index);
        }

        const std::string& interface_collapsed_name(int interface_collapsed_index) const override {
            return this->g.interface_collapsed_name(interface_collapsed_index);
        }

        const std::string& joint_collapsed_name(int joint_collapsed_index) const override {
            return this->g.joint_collapsed_name(joint_collapsed_index);
        }

        bool contains_interface_node(const std::string& name) const override {
            return this->g.contains_interface_node(name);
        }

        bool contains_total_node(const std::string& name) const override {
            return this->g.contains_total_node(name);
        }

        int add_interface_node(const std::string& node) override {
            return this->g.add_interface_node(node);
        }
        
        void remove_interface_node(int node_index) override {
            this->g.remove_interface_node(node_index);
        }
        
        void remove_interface_node(const std::string& node) override {
            this->g.remove_interface_node(node);
        }
        
        bool is_interface(int index) const override {
            return this->g.is_interface(index);
        }

        bool is_interface(const std::string& name) const override {
            return this->g.is_interface(name);
        }
        
        void set_interface(int index) override {
            this->g.set_interface(index);
            if (!this->m_cpds.empty()) {
                this->m_cpds[index] = CPD();
            }
        }

        void set_interface(const std::string& name) override {
            this->g.set_interface(name);
            if(!this->m_cpds.empty()) {
                this->m_cpds[this->index(name)] = CPD();
            }
        }

        virtual void set_node(int index) override {
            this->g.set_node(index);
            if (!this->m_cpds.empty())
                this->m_cpds[index] = static_cast<Derived&>(*this).create_cpd(this->g.name(index));
        }

        void set_node(const std::string& name) override {
            this->g.set_node(name);
            if(!this->m_cpds.empty())
                this->m_cpds[this->index(name)] = static_cast<Derived&>(*this).create_cpd(name);
        }

        bool can_have_cpd(int index) const {
            return this->is_valid(index) && !is_interface(index);
        }

        bool can_have_cpd(const std::string& name) const {
            return this->is_valid(this->index(name)) && !is_interface(name);
        }

        using BayesianNetworkImpl<Derived>::sample;
        DataFrame sample(const DataFrame& evidence, unsigned int seed, bool concat_evidence, bool ordered) const override;
    };

    template<typename Derived>
    bool BayesianNetworkImpl<Derived>::fitted() const {
        if (m_cpds.empty()) {
            return false;
        } else {
            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (static_cast<const Derived&>(*this).can_have_cpd(i) && !m_cpds[i].fitted()) {
                    return false;
                }
            }

            return true;
        }    
    }

    template<typename Derived>
    void BayesianNetworkImpl<Derived>::compatible_cpd(const CPD& cpd) const {
        if (!contains_node(cpd.variable())) {
            throw std::invalid_argument("CPD defined on variable which is not present in the model:\n" + cpd.ToString());
        }

        auto& evidence = cpd.evidence();

        for (auto& ev : evidence) {
            if constexpr (is_unconditional_bn_v<Derived>) {
                if (!contains_node(ev)) {
                    throw std::invalid_argument("Evidence variable " + ev + " is not present in the model:\n" + cpd.ToString());
                }
            } else if constexpr(is_conditional_bn_v<Derived>) {
                if (!this->contains_total_node(ev)) {
                    throw std::invalid_argument("Evidence variable " + ev + " is not present in the model:\n" + cpd.ToString());
                }
            } else {
                static_assert(util::always_false<Derived>, "Wrong BN Type");
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
                                    + "\nParents: " + parents_to_string(cpd.variable());
                throw std::invalid_argument(err);
            }
        }
    }

    template<typename Derived>
    void BayesianNetworkImpl<Derived>::add_cpds(const std::vector<CPD>& cpds) {
        for (auto& cpd : cpds) {
            static_cast<Derived*>(this)->compatible_cpd(cpd);
        }

        if (m_cpds.empty()) {
            std::unordered_map<std::string, typename std::vector<CPD>::const_iterator> map_index;
            for (auto it = cpds.begin(); it != cpds.end(); ++it) {
                if (map_index.count(it->variable()) == 1) {
                    throw std::invalid_argument("CPD for variable " + it->variable() + " is repeated.");
                }
                map_index[it->variable()] = it;
            }

            m_cpds.reserve(num_raw_nodes());

            for (int i = 0; i < num_raw_nodes(); ++i) {
                if (static_cast<Derived&>(*this).can_have_cpd(i)) {
                    const auto& node_name = name(i);
                    auto cpd_idx = map_index.find(node_name);

                    if (cpd_idx != map_index.end()) {
                        m_cpds.push_back(*(cpd_idx->second));
                    } else {
                        m_cpds.push_back(static_cast<Derived&>(*this).create_cpd(node_name));
                    }
                } else {
                    m_cpds.push_back(CPD());
                }
            }
        } else {
            for(auto& cpd : cpds) {
                if (static_cast<Derived&>(*this).can_have_cpd(cpd.variable())) {
                    auto idx = index(cpd.variable());
                    m_cpds[idx] = cpd;
                } else {
                    throw std::invalid_argument("CPD for node " + cpd.variable() + " not valid for Bayesian network.");
                }
            }
        }
    }

    template<typename Derived>
    bool BayesianNetworkImpl<Derived>::must_construct_cpd(const CPD& cpd) const {
        const auto& node = cpd.variable();
        const auto& cpd_evidence = cpd.evidence();
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
    void BayesianNetworkImpl<Derived>::fit(const DataFrame& df) {
        auto nraw_nodes = num_raw_nodes();
        if (m_cpds.empty()) {
            m_cpds.reserve(nraw_nodes);

            for (int i = 0; i < nraw_nodes; ++i) {
                if (static_cast<Derived&>(*this).can_have_cpd(i)) {
                    auto cpd = static_cast<Derived&>(*this).create_cpd(name(i));
                    m_cpds.push_back(std::move(cpd));
                    m_cpds.back().fit(df);
                } else {
                    m_cpds.push_back(CPD());
                }
            }
        } else {
            for (int i = 0; i < nraw_nodes; ++i) {
                if (static_cast<Derived&>(*this).can_have_cpd(i)) {
                    if (static_cast<const Derived&>(*this).must_construct_cpd(m_cpds[i])) {
                        m_cpds[i] = static_cast<Derived&>(*this).create_cpd(name(i));
                        m_cpds[i].fit(df);
                    } else if (!m_cpds[i].fitted()) {
                        m_cpds[i].fit(df);
                    }
                }
            }
        }
    }

    template<typename Derived>
    void BayesianNetworkImpl<Derived>::check_fitted() const {
        if (m_cpds.empty()) {
            throw py::value_error("Model not fitted.");
        } else {
            bool all_fitted = true;
            std::string err;
            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (static_cast<const Derived&>(*this).can_have_cpd(i) && !m_cpds[i].fitted()) {
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
    VectorXd BayesianNetworkImpl<Derived>::logl(const DataFrame& df) const {
        check_fitted();

        size_t i = 0;
        for (size_t i = 0; i < m_cpds.size() && !static_cast<const Derived&>(*this).can_have_cpd(i); ++i);

        VectorXd accum = m_cpds[i].logl(df);

        for (++i; i < m_cpds.size(); ++i) {
            if (static_cast<const Derived&>(*this).can_have_cpd(i))
                accum += m_cpds[i].logl(df);
        }

        return accum;
    }

    template<typename Derived>
    double BayesianNetworkImpl<Derived>::slogl(const DataFrame& df) const {
        check_fitted();
        
        double accum = 0;
        for (size_t i = 0; i < m_cpds.size(); ++i) {
            if (static_cast<const Derived&>(*this).can_have_cpd(i))
                accum += m_cpds[i].slogl(df);
        }
        return accum;
    }

    template<typename Derived>
    DataFrame BayesianNetworkImpl<Derived>::sample(int n, unsigned int seed, bool ordered) const {
        if constexpr (is_conditional_bn_v<Derived>) {
            if (this->num_interface_nodes() > 0)
                throw std::runtime_error("Can not sample from ConditionalBayesianNetwork "
                                         "if evidence is not provided for the interface nodes.");
        }

        check_fitted();

        DataFrame parents(n);

        auto top_sort = g.topological_sort();
        for (size_t i = 0; i < top_sort.size(); ++i) {
            auto idx = index(top_sort[i]);
            auto array = m_cpds[idx].sample(n, parents, seed);
            
            auto res = parents->AddColumn(i, top_sort[i], array);
            parents = DataFrame(std::move(res).ValueOrDie());
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
    std::unique_ptr<ConditionalBayesianNetworkBase> 
    BayesianNetworkImpl<Derived>::conditional_bn(const std::vector<std::string>& nodes,
                                                 const std::vector<std::string>& interface_nodes) const {
        auto new_dag = g.conditional_graph(nodes, interface_nodes);

        auto new_bn = std::make_unique<ConditionalBayesianNetwork<Derived::TYPE>>(std::move(new_dag));

        if (!m_cpds.empty()) {
            std::vector<CPD> cpds;
            cpds.reserve(nodes.size());

            for (const auto& name : new_bn->nodes()) {
                if (static_cast<const Derived&>(*this).can_have_cpd(name)) {
                    cpds.push_back(this->cpd(name));
                }
            }

            new_bn->add_cpds(cpds);
        } 

        return new_bn;
    }

    template<typename Derived>
    std::unique_ptr<ConditionalBayesianNetworkBase>
    BayesianNetworkImpl<Derived>::conditional_bn() const {
        if constexpr (is_conditional_bn_v<Derived>) {
            return this->clone();
        } else {
            std::vector<std::string> interface;
            return conditional_bn(this->nodes(), interface);
        }
    }
    
    template<typename Derived>
    std::unique_ptr<BayesianNetworkBase> BayesianNetworkImpl<Derived>::unconditional_bn() const {
        if constexpr (is_unconditional_bn_v<Derived>) {
            return this->clone();
        } else {
            auto new_dag = this->g.unconditional_graph();

            auto new_bn = std::make_unique<BayesianNetwork<Derived::TYPE>>(std::move(new_dag));

            if (!this->m_cpds.empty()) {
                std::vector<CPD> cpds;
                cpds.reserve(this->num_nodes());

                for (const auto& name : this->nodes()) {
                    auto idx = this->index(name);
                    cpds.push_back(this->m_cpds[idx]);
                }

                new_bn->add_cpds(cpds);
            }

            return new_bn;
        }
    }

    template<typename Derived>
    void BayesianNetworkImpl<Derived>::save(std::string name, bool include_cpd) const {
        m_include_cpd = include_cpd;
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(static_cast<const Derived*>(this)), file, 2);
        file.attr("close")();
    }

    template<typename Derived>
    py::tuple BayesianNetworkImpl<Derived>::__getstate__() const {
        auto g_tuple = g.__getstate__();

        auto extra_info = static_cast<const Derived&>(*this).__getstate_extra__();

        if (m_include_cpd && fitted()) {
            std::vector<py::tuple> cpds;
            cpds.reserve(g.num_nodes());

            for (size_t i = 0; i < m_cpds.size(); ++i) {
                if (static_cast<const Derived&>(*this).can_have_cpd(i))
                    cpds.push_back(m_cpds[i].__getstate__());
            }

            return py::make_tuple(g_tuple, true, cpds, extra_info);
        } {
            return py::make_tuple(g_tuple, false, py::make_tuple(), extra_info);
        }
    }

    template<typename Derived>
    Derived BayesianNetworkImpl<Derived>::__setstate__(py::tuple& t) {
        if (t.size() != 4)
            throw std::runtime_error("Not valid BayesianNetwork.");
        
        auto bn = Derived(DagClass::__setstate__(t[0].cast<py::tuple>()));

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

    template<typename Derived>
    DataFrame ConditionalBayesianNetworkImpl<Derived>::sample(const DataFrame& evidence,
                                                              unsigned int seed,
                                                              bool concat_evidence,
                                                              bool ordered) const {

        this->check_fitted();
        evidence.raise_has_columns(interface_nodes());
        
        DataFrame parents(evidence);

        auto top_sort = this->g.topological_sort();
        for (size_t i = 0; i < top_sort.size(); ++i) {
            if (!is_interface(top_sort[i])) {
                auto idx = this->index(top_sort[i]);
                auto array = this->m_cpds[idx].sample(evidence->num_rows(), parents, seed);

                auto res = parents->AddColumn(evidence->num_columns() + i, top_sort[i], array);
                parents = DataFrame(std::move(res).ValueOrDie());
            }
        }

        std::vector<Field_ptr> fields;
        std::vector<Array_ptr> columns;

        auto schema = parents->schema();
        if (ordered) {
            for (const auto& name : this->nodes()) {
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

    py::object load_model(const std::string& name);
    void requires_continuous_data(const DataFrame& df);
    void requires_discrete_data(const DataFrame& df);
}

#endif //PYBNESIAN_MODELS_BAYESIANNETWORK_HPP