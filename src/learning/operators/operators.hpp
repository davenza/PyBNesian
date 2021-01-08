#ifndef PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
#define PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP

#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>
#include <models/ConditionalBayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <util/validate_scores.hpp>
#include <util/vector.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
using VectorXb = Matrix<bool, Dynamic, 1>;

using models::BayesianNetwork, models::BayesianNetworkBase, models::SemiparametricBNBase;
using models::ConditionalBayesianNetworkBase;
using factors::FactorType;
using learning::scores::Score;
using util::ArcStringVector, util::FactorStringTypeVector;

namespace learning::operators {

    class OperatorType {
    public:
        enum Value : uint8_t
        {
            ADD_ARC,
            REMOVE_ARC,
            FLIP_ARC,
            CHANGE_NODE_TYPE
        };

        struct Hash
        {
            inline std::size_t operator ()(OperatorType const opset_type) const
            {
                return static_cast<std::size_t>(opset_type.value);
            }
        };

        using HashType = Hash;

        OperatorType() = default;
        constexpr OperatorType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(OperatorType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(OperatorType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::ADD_ARC:
                    return "AddArc";
                case Value::REMOVE_ARC:
                    return "RemoveArc";
                case Value::FLIP_ARC:
                    return "FlipArc";
                case Value::CHANGE_NODE_TYPE:
                    return "ChangeNodeType";
                default:
                    throw std::invalid_argument("Unreachable code in OperatorType.");
            }
        }

    private:
        Value value;
    };

    class Operator {
    public:
        Operator(double delta, OperatorType type) : m_delta(delta), m_type(type) {}
        virtual ~Operator() {};

        virtual void apply(BayesianNetworkBase& m) const = 0;
        virtual std::shared_ptr<Operator> opposite() const = 0;
        double delta() const { return m_delta; }
        OperatorType type() const { return m_type; }
        virtual std::shared_ptr<Operator> copy() const = 0;

        virtual std::string ToString() const = 0;

        bool operator==(const Operator& a) const;
        bool operator!=(const Operator& a) const {
            return !(*this == a);
        }
    private:
        double m_delta;
        OperatorType m_type;
    };

    class ArcOperator : public Operator {
    public:
        ArcOperator(std::string source, 
                    std::string target,
                    double delta,
                    OperatorType type) : Operator(delta, type), m_source(source), m_target(target) {}

        const std::string& source() const { return m_source; }
        const std::string& target() const { return m_target; }
    private:
        std::string m_source;
        std::string m_target;
    };

    class AddArc : public ArcOperator {
    public:
        AddArc(std::string source, 
               std::string target,
               double delta) :  ArcOperator(source, target, delta, OperatorType::ADD_ARC) {}

        void apply(BayesianNetworkBase& m) const override {
            m.add_arc(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() const override;
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<AddArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "AddArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) + ")";
        }   
    };

    class RemoveArc : public ArcOperator {
    public:
        RemoveArc(std::string source, 
                  std::string target,
                  double delta) : ArcOperator(source, target, delta, OperatorType::REMOVE_ARC) {}

        void apply(BayesianNetworkBase& m) const override {
            m.remove_arc(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() const override {
            return std::make_shared<AddArc>(this->source(), this->target(), -this->delta());
        }
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<RemoveArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "RemoveArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) + ")";
        }      
    };

    class FlipArc : public ArcOperator {
    public:
        FlipArc(std::string source, 
                std::string target,
                double delta) : ArcOperator(source, target, delta, OperatorType::FLIP_ARC) {}

        void apply(BayesianNetworkBase& m) const override {
            m.flip_arc(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() const override {
            return std::make_shared<FlipArc>(this->target(), this->source(), -this->delta());
        }
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<FlipArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "FlipArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) + ")";
        }  
    };

    class ChangeNodeType : public Operator {
    public:
        ChangeNodeType(std::string node,
                       FactorType new_node_type,
                       double delta) : Operator(delta, OperatorType::CHANGE_NODE_TYPE),
                                       m_node(node),
                                       m_new_node_type(new_node_type) {}

        const std::string& node() const { return m_node; }
        FactorType node_type() const { return m_new_node_type; }
        void apply(BayesianNetworkBase& m) const override {
            try {
                auto& spbn = dynamic_cast<SemiparametricBNBase&>(m);
                spbn.set_node_type(m_node, m_new_node_type);
            } catch (const std::bad_cast&) {
                throw std::invalid_argument("ChangeNodeType can only be applied to SemiparametricBN.");
            }
        }
        std::shared_ptr<Operator> opposite() const override {
            return std::make_shared<ChangeNodeType>(m_node, m_new_node_type.opposite(), -this->delta());
        }
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<ChangeNodeType>(m_node, m_new_node_type, this->delta());
        }
        std::string ToString() const override {
            return "ChangeNodeType(" + node() + " -> " + m_new_node_type.ToString() + "; Delta: " + std::to_string(this->delta()) + ")";
        }
    private:
        std::string m_node;
        FactorType m_new_node_type;
    };

    class HashOperator {
    public:
        inline std::size_t operator()(const std::shared_ptr<Operator>& op) const {
            switch(op->type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC:
                case OperatorType::FLIP_ARC: {
                    const auto& dwn_op = dynamic_cast<const ArcOperator&>(*op);
                    auto hasher = std::hash<std::string>{};
                    return (hasher(dwn_op.source()) << (op->type()+1)) ^ hasher(dwn_op.target());
                }
                break;
                case OperatorType::CHANGE_NODE_TYPE: {
                    const auto& dwn_op = dynamic_cast<const ChangeNodeType&>(*op);
                    auto hasher = std::hash<std::string>{};
                    return hasher(dwn_op.node()) * dwn_op.node_type();
                }
                break;
                default:
                    throw std::invalid_argument("[HashOperator] Wrong Operator.");
            }
        }
    };

    class OperatorPtrEqual {
    public:
        inline bool operator()(const std::shared_ptr<Operator>& lhs, 
                               const std::shared_ptr<Operator>& rhs) const {
            bool eq = (lhs->type() == rhs->type());
            if (!eq)
                return false; 


            switch(lhs->type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC:
                case OperatorType::FLIP_ARC: {
                    const auto& dwn_lhs = dynamic_cast<const ArcOperator&>(*lhs);
                    const auto& dwn_rhs = dynamic_cast<const ArcOperator&>(*rhs);
                    if ((dwn_lhs.source() == dwn_rhs.source()) && (dwn_lhs.target() == dwn_rhs.target()))
                        return true;
                    else
                        return false;
                }
                case OperatorType::CHANGE_NODE_TYPE: {
                    const auto& dwn_lhs = dynamic_cast<const ChangeNodeType&>(*lhs);
                    const auto& dwn_rhs = dynamic_cast<const ChangeNodeType&>(*rhs);
                    if ((dwn_lhs.node() == dwn_rhs.node()) && (dwn_lhs.node_type() == dwn_rhs.node_type()))
                        return true;
                    else
                        return false;
                }
                default:
                    throw std::invalid_argument("Unreachable code");
            }
        }
    };

    class OperatorTabuSet {
    public:
        OperatorTabuSet() : m_set() { }

        OperatorTabuSet(const OperatorTabuSet& other) : m_set() {
            for (const auto& op : other.m_set) {
                m_set.insert(op);
            }
        }

        OperatorTabuSet& operator=(const OperatorTabuSet& other) {
            clear();
            for (const auto& op : other.m_set) {
                m_set.insert(op);
            }

            return *this;
        }

        OperatorTabuSet(OperatorTabuSet&& other) : m_set(std::move(other.m_set)) {}
        OperatorTabuSet& operator=(OperatorTabuSet&& other) { m_set = std::move(other.m_set); return *this; }

        void insert(const std::shared_ptr<Operator>& op) {
            m_set.insert(op);
        }

        bool contains(const std::shared_ptr<Operator>& op) const {
            return m_set.count(op) > 0;
        }
        void clear() {
            m_set.clear();
        }
        bool empty() const {
            return m_set.empty();
        }
    private:
        using SetType = std::unordered_set<std::shared_ptr<Operator>, 
                                           HashOperator, 
                                           OperatorPtrEqual>;

        SetType m_set;
    };

    class OperatorSetType
    {
    public:
        enum Value : uint8_t
        {
            ARCS,
            NODE_TYPE
        };

        struct Hash
        {
            inline std::size_t operator ()(OperatorSetType const opset_type) const
            {
                return static_cast<std::size_t>(opset_type.value);
            }
        };

        using HashType = Hash;

        OperatorSetType() = default;
        constexpr OperatorSetType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(OperatorSetType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(OperatorSetType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::ARCS:
                    return "arcs";
                case Value::NODE_TYPE:
                    return "node_type";
                default:
                    throw std::invalid_argument("Unreachable code in OperatorSetType.");
            }
        }

    private:
        Value value;
    };

    class LocalScoreCache {
    public:
        LocalScoreCache() : m_local_score() {}
        LocalScoreCache(const BayesianNetworkBase& m) : m_local_score(m.num_nodes()) {}
        LocalScoreCache(const ConditionalBayesianNetworkBase& m) : m_local_score(m.num_nodes()) {}

        void cache_local_scores(const BayesianNetworkBase& model,
                                const Score& score) {
            if (m_local_score.rows() != model.num_nodes()) {
                m_local_score = VectorXd(model.num_nodes());
            }

            for (int i = 0; i < model.num_nodes(); ++i) {
                m_local_score(i) = score.local_score(model, i);
            }
        }

        void cache_local_scores(const ConditionalBayesianNetworkBase& model,
                                const Score& score) {
            if (m_local_score.rows() != model.num_nodes()) {
                m_local_score = VectorXd(model.num_nodes());
            }

            for (int i = 0; i < model.num_nodes(); ++i) {
                m_local_score(i) = score.local_score(model, model.index_from_collapsed(i));
            }
        }

        void update_local_score(const BayesianNetworkBase& model,
                                const Score& score,
                                int index) {
            m_local_score(index) = score.local_score(model, index);
        }

        void update_local_score(const ConditionalBayesianNetworkBase& model,
                                const Score& score,
                                int index) {
            auto collapsed = model.collapsed_from_index(index);
            m_local_score(collapsed) = score.local_score(model, index);
        }

        void update_local_score(const BayesianNetworkBase& model,
                                const Score& score,
                                const Operator& op) {
            switch(op.type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC: {
                    auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::FLIP_ARC: {
                    auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.source()));
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto& dwn_op = dynamic_cast<const ChangeNodeType&>(op);
                    update_local_score(model, score, model.index(dwn_op.node()));
                }
                    break;
            }
        }

        void update_local_score(const ConditionalBayesianNetworkBase& model,
                                const Score& score,
                                const Operator& op) {
            switch(op.type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC: {
                    auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::FLIP_ARC: {
                    auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.source()));
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto& dwn_op = dynamic_cast<const ChangeNodeType&>(op);
                    update_local_score(model, score, model.index(dwn_op.node()));
                }
                    break;
            }
        }

        double sum() {
            return m_local_score.sum();
        }

        double local_score(const BayesianNetworkBase&, int index) {
            return m_local_score(index);
        }

        double local_score(const ConditionalBayesianNetworkBase& model, int index) {
            return m_local_score(model.collapsed_from_index(index));
        }

        double local_score(const BayesianNetworkBase& model, const std::string& name) {
            return m_local_score(model.index(name));
        }

        double local_score(const ConditionalBayesianNetworkBase& model, const std::string& name) {
            return m_local_score(model.collapsed_index(name));
        }

    private:
        VectorXd m_local_score;
    };
    
    class OperatorSet {
    public:
        virtual ~OperatorSet() {}
        virtual void cache_scores(const BayesianNetworkBase&, const Score&) = 0;
        virtual std::shared_ptr<Operator> find_max(const BayesianNetworkBase&) const = 0;
        virtual std::shared_ptr<Operator> find_max(const BayesianNetworkBase&, const OperatorTabuSet&) const = 0;
        virtual void update_scores(const BayesianNetworkBase&, const Score&, const Operator&) = 0;

        virtual void cache_scores(const ConditionalBayesianNetworkBase&, const Score&) = 0;
        virtual std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase&) const = 0;
        virtual std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase&, const OperatorTabuSet&) const = 0;
        virtual void update_scores(const ConditionalBayesianNetworkBase&, const Score&, const Operator&) = 0;

        void set_local_score_cache(std::shared_ptr<LocalScoreCache> score_cache) {
            m_local_cache = score_cache;
        }

        virtual void set_arc_blacklist(const ArcStringVector&) = 0;
        virtual void set_arc_blacklist(const ArcSet&) = 0;
        virtual void set_arc_whitelist(const ArcStringVector&) = 0;
        virtual void set_arc_whitelist(const ArcSet&) = 0;
        virtual void set_max_indegree(int) = 0;
        virtual void set_type_whitelist(const FactorStringTypeVector&) = 0;
    protected:
        bool owns_local_cache() const {
            return m_local_cache.use_count() == 1;
        }

        template<typename M>
        void initialize_local_cache(M& model) {
            if (this->m_local_cache == nullptr) {
                auto lc = std::make_shared<LocalScoreCache>(model);
                this->set_local_score_cache(lc);
            }
        }

        void raise_uninitialized() const {
            if (m_local_cache == nullptr) {
                throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
            }
        }

        std::shared_ptr<LocalScoreCache> m_local_cache;
    };

    class ArcOperatorSet : public OperatorSet {
    public:
        ArcOperatorSet(ArcStringVector blacklist = ArcStringVector(), 
                       ArcStringVector whitelist = ArcStringVector(),
                       int indegree = 0) : delta(),
                                           valid_op(), 
                                           sorted_idx(),
                                           m_blacklist_names(blacklist),
                                           m_whitelist_names(whitelist),
                                           m_blacklist(),
                                           m_whitelist(),
                                           required_arclist_update(true),
                                           max_indegree(indegree) {}

        void cache_scores(const BayesianNetworkBase& model, const Score& score) override;
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override;
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override;
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(const BayesianNetworkBase& model) const;
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(const BayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const;
        void update_scores(const BayesianNetworkBase& model, const Score& score, const Operator& op) override;
        
        void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override;
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override;
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override;
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(const ConditionalBayesianNetworkBase& model) const;
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(const ConditionalBayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const;
        void update_scores(const ConditionalBayesianNetworkBase& model, const Score& score, const Operator& op) override;

        void update_incoming_arcs_scores(const BayesianNetworkBase& model, const Score& score, const std::string& dest_node);
        void update_incoming_arcs_scores(const ConditionalBayesianNetworkBase& model, const Score& score, const std::string& dest_node);

        void update_valid_ops(const BayesianNetworkBase& bn);
        void update_valid_ops(const ConditionalBayesianNetworkBase& bn);

        void set_arc_blacklist(const ArcStringVector& blacklist) override {
            m_blacklist_names = blacklist;
            required_arclist_update = true;
        }

        void set_arc_blacklist(const ArcSet& blacklist) override {
            m_blacklist = blacklist;
            required_arclist_update = true;
        }

        void set_arc_whitelist(const ArcStringVector& whitelist) override {
            m_whitelist_names = whitelist;
            required_arclist_update = true;
        }

        void set_arc_whitelist(const ArcSet& whitelist) override {
            m_whitelist = whitelist;
            required_arclist_update = true;
        }

        void set_max_indegree(int indegree) override {
            max_indegree = indegree;
        }
        
        void set_type_whitelist(const FactorStringTypeVector&) override {}
    private:
        MatrixXd delta;
        MatrixXb valid_op;
        mutable std::vector<int> sorted_idx;
        ArcStringVector m_blacklist_names;
        ArcStringVector m_whitelist_names;
        ArcSet m_blacklist;
        ArcSet m_whitelist;
        bool required_arclist_update;
        int max_indegree;
    };

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const BayesianNetworkBase& model) const {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            auto idx = *it;
            auto source = idx % model.num_nodes();
            auto dest = idx / model.num_nodes();

            if(model.has_arc(source, dest)) {
                return std::make_shared<RemoveArc>(model.name(source),
                                                   model.name(dest),
                                                   delta(source, dest));
            } else if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<FlipArc>(model.name(dest),
                                                 model.name(source),
                                                 delta(source, dest));
            } else if (model.can_add_arc(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<AddArc>(model.name(source),
                                                model.name(dest),
                                                delta(source, dest));
            }
        }

        return nullptr;
    }

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const ConditionalBayesianNetworkBase& model) const {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            auto idx = *it;
            auto source = idx % model.num_total_nodes();
            auto dest_collapsed = idx / model.num_total_nodes();
            auto dest = model.index_from_collapsed(dest_collapsed);

            auto d = delta(source, dest_collapsed);
            if(model.has_arc(source, dest)) {
                return std::make_shared<RemoveArc>(model.name(source), model.name(dest), d);
            }

            if (model.is_interface(source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                // If source is interface, the arc has a unique direction, and cannot produce cycles as source cannot have parents.
                return std::make_shared<AddArc>(model.name(source), model.name(dest), d);
            } else {                
                if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                    if constexpr (limited_indegree) {
                        if (model.num_parents(dest) >= max_indegree) {
                            continue;
                        }
                    }
                    return std::make_shared<FlipArc>(model.name(dest), model.name(source), d);
                } else if (model.can_add_arc(source, dest)) {
                    if constexpr (limited_indegree) {
                        if (model.num_parents(dest) >= max_indegree) {
                            continue;
                        }
                    }
                    return std::make_shared<AddArc>(model.name(source), model.name(dest), d);
                }
            }
        }

        return nullptr;
    }

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const BayesianNetworkBase& model,
                                                                const OperatorTabuSet& tabu_set) const {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            auto idx = *it;
            auto source = idx % model.num_nodes();
            auto dest = idx / model.num_nodes();

            if(model.has_arc(source, dest)) {
                std::shared_ptr<Operator> op = std::make_shared<RemoveArc>(model.name(source),
                                                                           model.name(dest),
                                                                           delta(source, dest));
                if (!tabu_set.contains(op))
                    return op;
            } else if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<FlipArc>(model.name(dest),
                                                                         model.name(source),
                                                                         delta(source, source));
                if (!tabu_set.contains(op))
                    return op;
            } else if (model.can_add_arc(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(model.name(source),
                                                                        model.name(dest),
                                                                        delta(source, dest));
                if (!tabu_set.contains(op))
                    return op;
            }
        }

        return nullptr;
    }

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const ConditionalBayesianNetworkBase& model,
                                                                const OperatorTabuSet& tabu_set) const {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            auto idx = *it;
            auto source = idx % model.num_total_nodes();
            auto dest_collapsed = idx / model.num_total_nodes();
            auto dest = model.index_from_collapsed(dest_collapsed);

            auto d = delta(source, dest_collapsed);

            if(model.has_arc(source, dest)) {
                std::shared_ptr<Operator> op = std::make_shared<RemoveArc>(model.name(source), model.name(dest), d);

                if (!tabu_set.contains(op))
                    return op;
                else continue;
            }

            if (model.is_interface(source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                // If source is interface, the arc has a unique direction, and cannot produce cycles as source cannot have parents.
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(model.name(source), model.name(dest), d);
                if (!tabu_set.contains(op))
                    return op;

            } else {                
                if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                    if constexpr (limited_indegree) {
                        if (model.num_parents(dest) >= max_indegree) {
                            continue;
                        }
                    }
                    std::shared_ptr<Operator> op =  std::make_shared<FlipArc>(model.name(dest), model.name(source), d);
                    if (!tabu_set.contains(op))
                        return op;
                } else if (model.can_add_arc(source, dest)) {
                    if constexpr (limited_indegree) {
                        if (model.num_parents(dest) >= max_indegree) {
                            continue;
                        }
                    }
                    std::shared_ptr<Operator> op = std::make_shared<AddArc>(model.name(source), model.name(dest), d);
                    if (!tabu_set.contains(op))
                        return op;
                }
            }
        }

        return nullptr;
    }

    class ChangeNodeTypeSet : public OperatorSet {
    public:
        ChangeNodeTypeSet(FactorStringTypeVector fv = FactorStringTypeVector()) : delta(),
                                                                                  valid_op(),
                                                                                  sorted_idx(),
                                                                                  m_type_whitelist(fv),
                                                                                  required_whitelist_update(true) {}

        void cache_scores(const BayesianNetworkBase& model, const Score& score) override;
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override;
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override;
        void update_scores(const BayesianNetworkBase& model, const Score& score, const Operator& op) override;

        void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override;
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override;
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override;
        void update_scores(const ConditionalBayesianNetworkBase& model, const Score& score, const Operator& op) override;

        void update_local_delta(const BayesianNetworkBase& model, const Score& score, const std::string& node) {
            update_local_delta(model, score, model.index(node));
        }

        void update_local_delta(const BayesianNetworkBase& model, const Score& score, int node_index) {
            auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
            FactorType type = spbn.node_type(node_index);
            auto parents = model.parent_indices(node_index);
            auto& spbn_score = dynamic_cast<const ScoreSPBN&>(score);
            delta(node_index) = spbn_score.local_score(type.opposite(), node_index, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(model, node_index);
        }

        void update_local_delta(const ConditionalBayesianNetworkBase& model, const Score& score, const std::string& node) {
            update_local_delta(model, score, model.index(node));
        }

        void update_local_delta(const ConditionalBayesianNetworkBase& model, const Score& score, int node_index) {
            auto collapsed = model.collapsed_from_index(node_index);
            auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
            FactorType type = spbn.node_type(node_index);
            auto parents = model.parent_indices(node_index);
            auto& spbn_score = dynamic_cast<const ScoreSPBN&>(score);
            delta(collapsed) = spbn_score.local_score(type.opposite(), node_index, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(model, node_index);
        }

        void update_whitelisted(const BayesianNetworkBase& model) {
            if (required_whitelist_update) {
                auto num_nodes = model.num_nodes();
                if (delta.rows() != num_nodes) {
                    delta = VectorXd(num_nodes);
                    valid_op = VectorXb(num_nodes);
                }

                auto val_ptr = valid_op.data();
                std::fill(val_ptr, val_ptr + model.num_nodes(), true);

                for (auto &node : m_type_whitelist) {
                    auto index = model.index(node.first);
                    delta(index) = std::numeric_limits<double>::lowest();;
                    valid_op(index) = false;
                }

                auto valid_ops = model.num_nodes() - m_type_whitelist.size();
                sorted_idx.clear();
                sorted_idx.reserve(valid_ops);
                for (auto i = 0; i < model.num_nodes(); ++i) {
                    if(valid_op(i))
                        sorted_idx.push_back(i);
                }
                required_whitelist_update = false;
            }
        }

        void update_whitelisted(const ConditionalBayesianNetworkBase& model) {
            if (required_whitelist_update) {
                auto num_nodes = model.num_nodes();
                if (delta.rows() != num_nodes) {
                    delta = VectorXd(num_nodes);
                    valid_op = VectorXb(num_nodes);
                }

                auto val_ptr = valid_op.data();
                std::fill(val_ptr, val_ptr + model.num_nodes(), true);

                auto indices = model.indices();

                for (auto &node : m_type_whitelist) {
                    auto index = model.collapsed_index(node.first);
                    delta(index) = std::numeric_limits<double>::lowest();
                    valid_op(index) = false;
                }

                auto valid_ops = model.num_nodes() - m_type_whitelist.size();
                sorted_idx.clear();
                sorted_idx.reserve(valid_ops);
                for (auto i = 0; i < model.num_nodes(); ++i) {
                    if(valid_op(i))
                        sorted_idx.push_back(i);
                }
                required_whitelist_update = false;
            }
        }

        void set_arc_blacklist(const ArcStringVector&) override {}
        void set_arc_blacklist(const ArcSet&) override {}
        void set_arc_whitelist(const ArcStringVector&) override {}
        void set_arc_whitelist(const ArcSet&) override {}
        void set_max_indegree(int) override {}
        void set_type_whitelist(const FactorStringTypeVector& type_whitelist) override {
            m_type_whitelist = type_whitelist;
            required_whitelist_update = true;
        }

    private:
        VectorXd delta;
        VectorXb valid_op;
        mutable std::vector<int> sorted_idx;
        FactorStringTypeVector m_type_whitelist;
        bool required_whitelist_update;
    };
    
    class OperatorPool : public OperatorSet {
    public:
        OperatorPool(std::vector<std::shared_ptr<OperatorSet>> op_sets) : m_op_sets(std::move(op_sets)) {
            if (m_op_sets.empty()) {
                throw std::invalid_argument("op_sets argument cannot be empty.");
            }
        }
        
        void cache_scores(const BayesianNetworkBase& model, const Score& score) override {
            return cache_scores<BayesianNetworkBase>(model, score);
        }
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override {
            return find_max<BayesianNetworkBase>(model);
        }
        std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override {
            return find_max<BayesianNetworkBase>(model, tabu_set);
        }
        void update_scores(const BayesianNetworkBase& model, const Score& score, const Operator& op) override {
            return update_scores<BayesianNetworkBase>(model, score, op);
        }

        void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override {
            return cache_scores<ConditionalBayesianNetworkBase>(model, score);
        }
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override {
            return find_max<ConditionalBayesianNetworkBase>(model);
        }
        std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model, const OperatorTabuSet& tabu_set) const override {
            return find_max<ConditionalBayesianNetworkBase>(model, tabu_set);
        }
        void update_scores(const ConditionalBayesianNetworkBase& model, const Score& score, const Operator& op) override {
            return update_scores<ConditionalBayesianNetworkBase>(model, score, op);
        }

        template<typename M>
        void cache_scores(const M& model, const Score& score);
        template<typename M>
        std::shared_ptr<Operator> find_max(const M& model) const;
        template<typename M>
        std::shared_ptr<Operator> find_max(const M& model, const OperatorTabuSet& tabu_set) const;
        template<typename M>
        void update_scores(const M& model, const Score& score, const Operator& op);

               
        void set_arc_blacklist(const ArcStringVector& blacklist) override {
            for(auto& opset : m_op_sets) {
                opset->set_arc_blacklist(blacklist);
            }
        }

        void set_arc_blacklist(const ArcSet& blacklist) override {
            for(auto& opset : m_op_sets) {
                opset->set_arc_blacklist(blacklist);
            }
        }

        void set_arc_whitelist(const ArcStringVector& whitelist) override {
            for(auto& opset : m_op_sets) {
                opset->set_arc_whitelist(whitelist);
            }
        }

        void set_arc_whitelist(const ArcSet& whitelist) override {
            for(auto& opset : m_op_sets) {
                opset->set_arc_whitelist(whitelist);
            }
        }

        void set_max_indegree(int indegree) override {
            for(auto& opset : m_op_sets) {
                opset->set_max_indegree(indegree);
            }
        }

        void set_type_whitelist(const FactorStringTypeVector& type_whitelist) override {
            for(auto& opset : m_op_sets) {
                opset->set_type_whitelist(type_whitelist);
            }
        }
    private:
        std::vector<std::shared_ptr<OperatorSet>> m_op_sets;
    };

    template<typename M>
    void OperatorPool::cache_scores(const M& model, const Score& score) {
        initialize_local_cache(model);
        m_local_cache->cache_local_scores(model, score);

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores(model, score);
        }
    }

    template<typename M>
    std::shared_ptr<Operator> OperatorPool::find_max(const M& model) const {
        raise_uninitialized();

        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto& op_set : m_op_sets) {
            auto new_op = op_set->find_max(model);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    template<typename M>
    std::shared_ptr<Operator> OperatorPool::find_max(const M& model, const OperatorTabuSet& tabu_set) const {
        raise_uninitialized();

        if (tabu_set.empty())
            return find_max(model);
        
        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto& op_set : m_op_sets) {
            auto new_op = op_set->find_max(model, tabu_set);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    template<typename M>
    void OperatorPool::update_scores(const M& model, const Score& score, const Operator& op) {
        raise_uninitialized();

        m_local_cache->update_local_score(model, score, op);
        for (auto& op_set : m_op_sets) {
            op_set->update_scores(model, score, op);
        }
    }

}

#endif //PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
