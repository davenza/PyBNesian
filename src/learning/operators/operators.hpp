#ifndef PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
#define PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP

#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <util/validate_scores.hpp>
#include <util/vector.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
using VectorXb = Matrix<bool, Dynamic, 1>;

using models::BayesianNetwork, models::BayesianNetworkBase, models::SemiparametricBNBase;
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

        virtual void apply(BayesianNetworkBase& m) = 0;
        virtual std::shared_ptr<Operator> opposite() = 0;
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

        void apply(BayesianNetworkBase& m) override {
            m.add_arc(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() override;
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

        void apply(BayesianNetworkBase& m) override {
            m.remove_arc(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() override {
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

        void apply(BayesianNetworkBase& m) override {
            m.remove_arc(this->source(), this->target());
            m.add_arc(this->target(), this->source());
        }
        std::shared_ptr<Operator> opposite() override {
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

        // static_assert(util::is_semiparametricbn_v<Model>, "ChangeNodeType operator can only be used with a SemiparametricBN.")
        ChangeNodeType(std::string node,
                       FactorType new_node_type,
                       double delta) : Operator(delta, OperatorType::CHANGE_NODE_TYPE),
                                       m_node(node),
                                       m_new_node_type(new_node_type) {}

        const std::string& node() const { return m_node; }
        FactorType node_type() const { return m_new_node_type; }
        void apply(BayesianNetworkBase& m) override {
            try {
                auto& spbn = dynamic_cast<SemiparametricBNBase&>(m);
                spbn.set_node_type(m_node, m_new_node_type);
            } catch (const std::bad_cast&) {
                throw std::invalid_argument("ChangeNodeType can only be applied to SemiparametricBN.");
            }
        }
        std::shared_ptr<Operator> opposite() override {
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
        inline std::size_t operator()(Operator* const op) const {
            switch(op->type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC:
                case OperatorType::FLIP_ARC: {
                    auto dwn_op = dynamic_cast<ArcOperator*>(op);
                    auto hasher = std::hash<std::string>{};
                    return (hasher(dwn_op->source()) << (op->type()+1)) ^ hasher(dwn_op->target());
                }
                break;
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto dwn_op = dynamic_cast<ChangeNodeType*>(op);
                    auto hasher = std::hash<std::string>{};

                    return hasher(dwn_op->node()) * dwn_op->node_type();
                }
                break;
                default:
                    throw std::invalid_argument("[HashOperator] Wrong Operator.");
            }
        }
    };

    class OperatorPtrEqual {
    public:
        inline bool operator()(const Operator* lhs, const Operator* rhs) const {
            bool eq = (lhs->type() == rhs->type());
            if (!eq)
                return false; 


            switch(lhs->type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC:
                case OperatorType::FLIP_ARC: {
                    auto dwn_lhs = dynamic_cast<const ArcOperator*>(lhs);
                    auto dwn_rhs = dynamic_cast<const ArcOperator*>(rhs);
                    if ((dwn_lhs->source() == dwn_rhs->source()) && (dwn_lhs->target() == dwn_rhs->target()))
                        return true;
                    else
                        return false;
                }
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto dwn_lhs = dynamic_cast<const ChangeNodeType*>(lhs);
                    auto dwn_rhs = dynamic_cast<const ChangeNodeType*>(rhs);
                    if ((dwn_lhs->node() == dwn_rhs->node()) && (dwn_lhs->node_type() == dwn_rhs->node_type()))
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
        OperatorTabuSet() : m_map() { }

        OperatorTabuSet(const OperatorTabuSet& other) : m_map() {
            for (auto& pair : other.m_map) {
                // auto copy = pair.second->copy();
                m_map.insert({pair.first, pair.second});
            }
        }

        OperatorTabuSet& operator=(const OperatorTabuSet& other) {
            clear();
            for (auto& pair : other.m_map) {
                // auto copy = pair.second->copy();
                m_map.insert({pair.first, pair.second});
            }

            return *this;
        }

        OperatorTabuSet(OperatorTabuSet&& other) : m_map(std::move(other.m_map)) {}
        OperatorTabuSet& operator=(OperatorTabuSet&& other) { m_map = std::move(other.m_map); return *this; }

        void insert(std::shared_ptr<Operator> op) {
            m_map.insert({op.get(), op});
        }
        bool contains(std::shared_ptr<Operator>& op) const {
            return m_map.count(op.get()) > 0;
        }
        void clear() {
            m_map.clear();
        }
        bool empty() const {
            return m_map.empty();
        }
    private:
        using MapType = std::unordered_map<Operator*, 
                                           std::shared_ptr<Operator>, 
                                           HashOperator, 
                                           OperatorPtrEqual>;

        MapType m_map;
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
        LocalScoreCache(BayesianNetworkBase& m) : m_local_score(m.num_nodes()) {}

        void cache_local_scores(BayesianNetworkBase& model, Score& score) {
            if (m_local_score.rows() != model.num_nodes()) {
                m_local_score = VectorXd(model.num_nodes());
            }

            for (int i = 0; i < model.num_nodes(); ++i) {
                m_local_score(i) = score.local_score(model, i);
            }
        }

        void update_local_score(BayesianNetworkBase& model, Score& score, int index) {
            m_local_score(index) = score.local_score(model, index);
        }

        void update_local_score(BayesianNetworkBase& model, Score& score, Operator& op) {
            switch(op.type()) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC: {
                    auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::FLIP_ARC: {
                    auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                    update_local_score(model, score, model.index(dwn_op.source()));
                    update_local_score(model, score, model.index(dwn_op.target()));
                }
                    break;
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                    update_local_score(model, score, model.index(dwn_op.node()));
                }
                    break;
            }
        }

        double sum() {
            return m_local_score.sum();
        }

        double local_score(int index) {
            return m_local_score(index);
        }

    private:
        VectorXd m_local_score;
    };
    
    class OperatorSet {
    public:
        virtual ~OperatorSet() {}
        virtual void cache_scores(BayesianNetworkBase&, Score&) = 0;
        virtual std::shared_ptr<Operator> find_max(BayesianNetworkBase&) = 0;
        virtual std::shared_ptr<Operator> find_max(BayesianNetworkBase&, OperatorTabuSet&) = 0;
        virtual void update_scores(BayesianNetworkBase&, Score&, Operator&) = 0;

        void set_local_score_cache(std::shared_ptr<LocalScoreCache>& score_cache) {
            m_local_cache = score_cache;
        }

        virtual void set_arc_blacklist(const ArcStringVector&) = 0;
        virtual void set_arc_whitelist(const ArcStringVector&) = 0;
        virtual void set_max_indegree(int) = 0;
        virtual void set_type_whitelist(const FactorStringTypeVector&) = 0;
    protected:
        bool owns_local_cache() {
            return m_local_cache.use_count() == 1;
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
                                           m_blacklist(blacklist),
                                           m_whitelist(whitelist),
                                           required_arclist_update(true),
                                           max_indegree(indegree) {}

        void cache_scores(BayesianNetworkBase& model, Score& score) override;

        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model) override;
        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) override;
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(BayesianNetworkBase& model);
        template<bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(BayesianNetworkBase& model, OperatorTabuSet& tabu_set);

        void update_scores(BayesianNetworkBase& model, Score& score, Operator& op) override;
        void update_node_arcs_scores(BayesianNetworkBase& model, Score& score, const std::string& dest_node);

        void update_listed_arcs(BayesianNetworkBase& bn);

        void set_arc_blacklist(const ArcStringVector& blacklist) override {
            m_blacklist = blacklist;
            required_arclist_update = true;
        }
        void set_arc_whitelist(const ArcStringVector& whitelist) override {
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
        std::vector<int> sorted_idx;
        ArcStringVector m_blacklist;
        ArcStringVector m_whitelist;
        bool required_arclist_update;
        int max_indegree;
    };

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(BayesianNetworkBase& model) {
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
                return std::make_shared<RemoveArc>(model.name(source), model.name(dest), delta(source, dest));
            } else if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<FlipArc>(model.name(dest), model.name(source), delta(dest, source));
            } else if (model.can_add_arc(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<AddArc>(model.name(source), model.name(dest), delta(source, dest));
            }
        }

        return nullptr;
    }

    template<bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
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
                std::shared_ptr<Operator> op = std::make_shared<RemoveArc>(model.name(source), model.name(dest), delta(source, dest));
                if (!tabu_set.contains(op))
                    return op;
            } else if (model.has_arc(dest, source) && model.can_flip_arc(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<FlipArc>(model.name(dest), model.name(source), delta(dest, source));
                if (!tabu_set.contains(op))
                    return op;
            } else if (model.can_add_arc(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(model.name(source), model.name(dest), delta(source, dest));
                if (!tabu_set.contains(op))
                    return op;
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

        void cache_scores(BayesianNetworkBase& model, Score& score) override;
        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model) override;
        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) override;
        void update_scores(BayesianNetworkBase& model, Score& score, Operator& op) override;

        void update_local_delta(BayesianNetworkBase& model, Score& score, const std::string& node) {
            update_local_delta(model, score, model.index(node));
        }

        void update_local_delta(BayesianNetworkBase& model, Score& score, int node_index) {
            auto& spbn = dynamic_cast<SemiparametricBNBase&>(model);
            FactorType type = spbn.node_type(node_index);
            auto parents = model.parent_indices(node_index);
            auto& spbn_score = dynamic_cast<ScoreSPBN&>(score);
            delta(node_index) = spbn_score.local_score(type.opposite(), node_index, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(node_index);
        }

        void update_whitelisted(BayesianNetworkBase& model) {
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
                    delta(indices[node.first]) = std::numeric_limits<double>::lowest();;
                    valid_op(indices[node.first]) = false;
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
        void set_arc_whitelist(const ArcStringVector&) override {}
        void set_max_indegree(int) override {}
        void set_type_whitelist(const FactorStringTypeVector& type_whitelist) override {
            m_type_whitelist = type_whitelist;
            required_whitelist_update = true;
        }

    private:
        VectorXd delta;
        VectorXb valid_op;
        std::vector<int> sorted_idx;
        FactorStringTypeVector m_type_whitelist;
        bool required_whitelist_update;
    };
    
    class OperatorPool {
    public:
        OperatorPool(std::shared_ptr<Score>& score, 
                     std::vector<std::shared_ptr<OperatorSet>> op_sets) : m_score(score),
                                                                          local_cache(std::make_shared<LocalScoreCache>()),
                                                                          m_op_sets(std::move(op_sets)) {
            if (m_op_sets.empty()) {
                throw std::invalid_argument("op_sets argument cannot be empty.");
            }
            
            for (auto& op_set : m_op_sets) {
                op_set->set_local_score_cache(local_cache);
            }
        }
        
        void cache_scores(BayesianNetworkBase& model);
        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model);
        std::shared_ptr<Operator> find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set);
        void update_scores(BayesianNetworkBase& model, Operator& op);
               
        double score() {
            return local_cache->sum();
        }

        double score(BayesianNetworkBase& model) {
            return m_score->score(model);
        }

        void set_arc_blacklist(const ArcStringVector& blacklist) {
            for(auto& opset : m_op_sets) {
                opset->set_arc_blacklist(blacklist);
            }
        }
        void set_arc_whitelist(const ArcStringVector& whitelist) {
            for(auto& opset : m_op_sets) {
                opset->set_arc_whitelist(whitelist);
            }
        }
        void set_max_indegree(int indegree) {
            for(auto& opset : m_op_sets) {
                opset->set_max_indegree(indegree);
            }
        }
        void set_type_whitelist(const FactorStringTypeVector& type_whitelist) {
            for(auto& opset : m_op_sets) {
                opset->set_type_whitelist(type_whitelist);
            }
        }
    private:
        std::shared_ptr<Score> m_score;
        std::shared_ptr<LocalScoreCache> local_cache;
        std::vector<std::shared_ptr<OperatorSet>> m_op_sets;
    };
}

#endif //PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
