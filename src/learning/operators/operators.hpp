#ifndef PGM_DATASET_OPERATORS_HPP
#define PGM_DATASET_OPERATORS_HPP

#include <queue>
#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <util/util_types.hpp>
#include <util/validate_scores.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
using VectorXb = Matrix<bool, Dynamic, 1>;

using graph::AdjListDag;
using models::BayesianNetwork, models::BayesianNetworkBase;
using factors::FactorType;
using learning::scores::Score;
using util::ArcVector, util::FactorTypeVector;

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
                    throw std::invalid_argument("Unreachable code in OperatorSetType.");
            }
        }

    private:
        Value value;
    };


    class Operator {
    public:
        Operator(double delta, OperatorType type) : m_delta(delta), m_type(type) {}

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
            m.add_edge(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() override;
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<AddArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "AddArc(" + this->source() + " -> " + this->target() + "; " + std::to_string(this->delta()) + ")";
        }   
    };

    class RemoveArc : public ArcOperator {
    public:
        RemoveArc(std::string source, 
                  std::string target,
                  double delta) : ArcOperator(source, target, delta, OperatorType::REMOVE_ARC) {}
        
        void apply(BayesianNetworkBase& m) override {
            m.remove_edge(this->source(), this->target());
        }
        std::shared_ptr<Operator> opposite() override {
            return std::make_shared<AddArc>(this->source(), this->target(), -this->delta());
        }
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<RemoveArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "RemoveArc(" + this->source() + " -> " + this->target() + "; " + std::to_string(this->delta()) + ")";
        }      
    };

    class FlipArc : public ArcOperator {
    public:
        FlipArc(std::string source, 
                std::string target,
                double delta) : ArcOperator(source, target, delta, OperatorType::FLIP_ARC) {}

        void apply(BayesianNetworkBase& m) override {
            m.remove_edge(this->source(), this->target());
            m.add_edge(this->target(), this->source());
        }
        std::shared_ptr<Operator> opposite() override {
            return std::make_shared<FlipArc>(this->target(), this->source(), -this->delta());
        }
        std::shared_ptr<Operator> copy() const override {
            return std::make_shared<FlipArc>(this->source(), this->target(), this->delta());
        }
        std::string ToString() const override {
            return "FlipArc(" + this->source() + " -> " + this->target() + "; " + std::to_string(this->delta()) + ")";
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
            return "ChangeNodeType(" + node() + " -> " + m_new_node_type.ToString() + "; " + std::to_string(this->delta()) + ")";
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

        template<typename Model>
        LocalScoreCache(Model& m) : m_local_score(m.num_nodes()) {}


        template<typename Model>
        void cache_local_scores(Model& model, Score& score) {
            for (int i = 0; i < model.num_nodes(); ++i) {
                m_local_score(i) = score.local_score(model, i);
            }
        }

        template<typename Model>
        void update_local_score(Model& model, Score& score, int index) {
            m_local_score(index) = score.local_score(model, index);
        }

        template<typename Model>
        void update_local_score(Model& model, Score& score, Operator& op) {
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
    
    template<typename... Models>
    class OperatorSetInterface {};

    template<typename Model>
    class OperatorSetInterface<Model> {
    public:
        virtual void cache_scores(Model& m) {
            throw std::invalid_argument("OperatorSet::cache_scores() not implemented for model " + m.type().ToString() + ".");
        }
        virtual std::shared_ptr<Operator> find_max(Model& m) {
            throw std::invalid_argument("OperatorSet::find_max() not implemented for model " + m.type().ToString() + ".");
        }
        virtual std::shared_ptr<Operator> find_max(Model& m, OperatorTabuSet&) {
            throw std::invalid_argument("OperatorSet::find_max() not implemented for model " + m.type().ToString() + ".");
        }
        virtual void update_scores(Model& m, Operator&) {
            throw std::invalid_argument("OperatorSet::update_scores() not implemented for model " + m.type().ToString() + ".");
        }
    };

    template<typename Model, typename... Models>
    class OperatorSetInterface<Model, Models...> : public OperatorSetInterface<Models...> {
    public:
        using Base = OperatorSetInterface<Models...>;
        using Base::cache_scores;
        using Base::find_max;
        using Base::update_scores;
        virtual void cache_scores(Model&) {
            throw std::invalid_argument("OperatorSet::cache_scores() not implemented.");
        }
        virtual std::shared_ptr<Operator> find_max(Model&) {
            throw std::invalid_argument("OperatorSet::find_max() not implemented.");
        }
        virtual std::shared_ptr<Operator> find_max(Model&, OperatorTabuSet&) {
            throw std::invalid_argument("OperatorSet::find_max() not implemented.");
        }
        virtual void update_scores(Model&, Operator&) {
            throw std::invalid_argument("OperatorSet::update_scores() not implemented.");
        }
    };

    class OperatorSet : public OperatorSetInterface<GaussianNetwork<>, 
                                                    GaussianNetwork<AdjListDag>,
                                                    SemiparametricBN<>,
                                                    SemiparametricBN<AdjListDag>>
    {
    public:
        void set_local_score_cache(std::shared_ptr<LocalScoreCache>& score_cache) {
            m_local_cache = score_cache;
        }

        virtual void set_arc_blacklist(const ArcVector&) = 0;
        virtual void set_arc_whitelist(const ArcVector&) = 0;
        virtual void set_max_indegree(int) = 0;
        virtual void set_type_whitelist(const FactorTypeVector&) = 0;
    protected:
        bool owns_local_cache() {
            return m_local_cache.use_count() == 1;
        }
        std::shared_ptr<LocalScoreCache> m_local_cache;
    };

    template<typename Derived, typename... Models>
    class OperatorSetImpl {};

    template<typename Derived, typename Model>
    class OperatorSetImpl<Derived, Model>  : public OperatorSet {
    public:
        void cache_scores(Model& m) override {
            static_cast<Derived*>(this)->cache_scores(m);
        }
        std::shared_ptr<Operator> find_max(Model& m) override {
            return static_cast<Derived*>(this)->find_max(m);
        }
        std::shared_ptr<Operator> find_max(Model& m, OperatorTabuSet& tabu) override {
            return static_cast<Derived*>(this)->find_max(m, tabu);
        }
        void update_scores(Model& m, Operator& op) override {
            static_cast<Derived*>(this)->update_scores(m, op);
        }
    };

    template<typename Derived, typename Model, typename... Models>
    class OperatorSetImpl<Derived, Model, Models...> : 
                public OperatorSetImpl<Derived,Models...> {
    public:
        void cache_scores(Model& m) override {
            static_cast<Derived*>(this)->cache_scores(m);
        }
        std::shared_ptr<Operator> find_max(Model& m) override {
            return static_cast<Derived*>(this)->find_max(m);
        }
        std::shared_ptr<Operator> find_max(Model& m, OperatorTabuSet& tabu) override {
            return static_cast<Derived*>(this)->find_max(m, tabu);
        }
        void update_scores(Model& m, Operator& op) override {
            static_cast<Derived*>(this)->update_scores(m, op);
        }
    };

    class ArcOperatorSet : public OperatorSetImpl<ArcOperatorSet,
                                                  GaussianNetwork<>,
                                                  GaussianNetwork<AdjListDag>,
                                                  SemiparametricBN<>,
                                                  SemiparametricBN<AdjListDag>>
    {
    public:
        ArcOperatorSet(std::shared_ptr<Score>& score, 
                       ArcVector blacklist = ArcVector(), 
                       ArcVector whitelist = ArcVector(),
                       int indegree = 0) : m_score(score),
                                           delta(),
                                           valid_op(), 
                                           sorted_idx(),
                                           m_blacklist(blacklist),
                                           m_whitelist(whitelist),
                                           required_arclist_update(true),
                                           max_indegree(indegree) {}

        template<typename Model>
        void cache_scores(Model& model);

        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model);
        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model, OperatorTabuSet& tabu_set);
        template<typename Model, bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(Model& model);
        template<typename Model, bool limited_indigree>
        std::shared_ptr<Operator> find_max_indegree(Model& model, OperatorTabuSet& tabu_set);
        template<typename Model>
        void update_scores(Model& model, Operator& op);
        template<typename Model>
        void update_node_arcs_scores(Model& model, const std::string& dest_node);

        void update_listed_arcs(BayesianNetworkBase& bn);

        void set_arc_blacklist(const ArcVector& blacklist) override {
            m_blacklist = blacklist;
            required_arclist_update = true;
        }
        void set_arc_whitelist(const ArcVector& whitelist) override {
            m_whitelist = whitelist;
            required_arclist_update = true;
        }
        void set_max_indegree(int indegree) override {
            max_indegree = indegree;
        }
        void set_type_whitelist(const FactorTypeVector&) override {}
    private:
        std::shared_ptr<Score> m_score;
        MatrixXd delta;
        MatrixXb valid_op;
        std::vector<int> sorted_idx;
        ArcVector m_blacklist;
        ArcVector m_whitelist;
        bool required_arclist_update;
        int max_indegree;
    };

    template<typename Model>
    void ArcOperatorSet::cache_scores(Model& model) {
        if (!util::compatible_score<Model>(m_score->type())) {
            throw std::invalid_argument("Invalid score " + m_score->ToString() + " for model type " + model.type().ToString() + ".");
        }

        update_listed_arcs(model);

        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->cache_local_scores(model, *m_score);
        } else if (this->owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, *m_score);
        }

        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {
                    if (model.has_edge(source, dest)) {            
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = m_score->local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - 
                                    this->m_local_cache->local_score(dest);
                        delta(source, dest) = d;
                    } else if (model.has_edge(dest, source)) {
                        auto new_parents_source = model.parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = m_score->local_score(model, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   m_score->local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - this->m_local_cache->local_score(source) - this->m_local_cache->local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = m_score->local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                    - this->m_local_cache->local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }
    }

    template<typename Model>
    std::shared_ptr<Operator> ArcOperatorSet::find_max(Model& model) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }

        if (max_indegree > 0)
            return find_max_indegree<Model, true>(model);
        else
            return find_max_indegree<Model, false>(model);
    }

    template<typename Model>
    std::shared_ptr<Operator> ArcOperatorSet::find_max(Model& model, OperatorTabuSet& tabu_set) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }

        if (max_indegree > 0)
            return find_max_indegree<Model, true>(model, tabu_set);
        else
            return find_max_indegree<Model, false>(model, tabu_set);
    }


    template<typename Model, bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(Model& model) {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % model.num_nodes();
            auto dest = idx / model.num_nodes();

            if(model.has_edge(source, dest)) {
                return std::make_shared<RemoveArc>(model.name(source), model.name(dest), delta(source, dest));
            } else if (model.has_edge(dest, source) && model.can_flip_edge(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<FlipArc>(model.name(dest), model.name(source), delta(dest, source));
            } else if (model.can_add_edge(source, dest)) {
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

    template<typename Model, bool limited_indegree>
    std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(Model& model,  OperatorTabuSet& tabu_set) {
        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % model.num_nodes();
            auto dest = idx / model.num_nodes();

            if(model.has_edge(source, dest)) {
                std::shared_ptr<Operator> op = std::make_shared<RemoveArc>(model.name(source), model.name(dest), delta(source, dest));
                if (!tabu_set.contains(op))
                    return std::move(op);
            } else if (model.has_edge(dest, source) && model.can_flip_edge(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<FlipArc>(model.name(dest), model.name(source), delta(dest, source));
                if (!tabu_set.contains(op))
                    return std::move(op);
            } else if (model.can_add_edge(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(model.name(source), model.name(dest), delta(source, dest));
                if (!tabu_set.contains(op))
                    return std::move(op);
            }
        }

        return nullptr;
    }

    template<typename Model>
    void ArcOperatorSet::update_scores(Model& model, Operator& op) {
        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->update_local_score(model, *m_score, op);
        } else if(this->owns_local_cache()) {
            this->m_local_cache->update_local_score(model, *m_score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, dwn_op.source());
                update_node_arcs_scores(model, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                update_node_arcs_scores(model, dwn_op.node());
            }
                break;
        }
    }   

    template<typename Model>
    void ArcOperatorSet::update_node_arcs_scores(Model& model, const std::string& dest_node) {

        auto dest_idx = model.index(dest_node);
        auto parents = model.parent_indices(dest_idx);
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {

                if (model.has_edge(i, dest_idx)) {
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = m_score->local_score(model, dest_idx, parents.begin(), parents.end() - 1) - 
                               this->m_local_cache->local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    auto new_parents_i = model.parent_indices(i);
                    new_parents_i.push_back(dest_idx);

                    delta(dest_idx, i) = d + m_score->local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                            - this->m_local_cache->local_score(i);
                } else if (model.has_edge(dest_idx, i)) {
                    auto new_parents_i = model.parent_indices(i);
                    std::iter_swap(std::find(new_parents_i.begin(), new_parents_i.end(), dest_idx), new_parents_i.end() - 1);
                        
                    parents.push_back(i);
                    double d = m_score->local_score(model, i, new_parents_i.begin(), new_parents_i.end() - 1) + 
                                m_score->local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(i) - this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = m_score->local_score(model, dest_idx, parents.begin(), parents.end()) - this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }
            }
        }
    }

    class ChangeNodeTypeSet : public OperatorSetImpl<ChangeNodeTypeSet,
                                                     SemiparametricBN<>,
                                                     SemiparametricBN<AdjListDag>> {
    public:
        ChangeNodeTypeSet(std::shared_ptr<Score>& score, 
                          FactorTypeVector fv = FactorTypeVector()) : m_score(score),
                                                                        delta(),
                                                                        valid_op(),
                                                                        sorted_idx(),
                                                                        m_type_whitelist(fv),
                                                                        required_whitelist_update(true) {}

        template<typename Model>
        void cache_scores(Model& model);
        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model);
        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model, OperatorTabuSet& tabu_set);
        template<typename Model>
        void update_scores(Model& model, Operator& op);

        template<typename Model>
        void update_local_delta(Model& model, const std::string& node) {
            update_local_delta(model, model.index(node));
        }

        template<typename Model>
        void update_local_delta(Model& model, int node_index) {
            FactorType type = model.node_type(node_index);
            auto parents = model.parent_indices(node_index);
            delta(node_index) = m_score->local_score(type.opposite(), node_index, parents.begin(), parents.end()) 
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

        void set_arc_blacklist(const ArcVector&) override {}
        void set_arc_whitelist(const ArcVector&) override {}
        void set_max_indegree(int) override {}
        void set_type_whitelist(const FactorTypeVector& type_whitelist) override {
            m_type_whitelist = type_whitelist;
            required_whitelist_update = true;
        }

    private:
        std::shared_ptr<Score> m_score;
        VectorXd delta;
        VectorXb valid_op;
        std::vector<int> sorted_idx;
        FactorTypeVector m_type_whitelist;
        bool required_whitelist_update;
    };

    template<typename Model>
    void ChangeNodeTypeSet::cache_scores(Model& model) {
        if (!util::compatible_score<Model>(m_score->type())) {
            throw std::invalid_argument("Invalid score " + m_score->ToString() + " for model type " + model.type().ToString() + ".");
        }
        
        update_whitelisted(model);

        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->cache_local_scores(model, *m_score);
        } else if (this->owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, *m_score);
        }

        for(auto i = 0; i < model.num_nodes(); ++i) {
            if(valid_op(i)) {
                update_local_delta(model, i);
            }
        }
    }

    template<typename Model>
    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(Model& model) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }
        auto delta_ptr = delta.data();
        auto max_element = std::max_element(delta_ptr, delta_ptr + model.num_nodes());
        int idx_max = std::distance(delta_ptr, max_element);
        auto node_type = model.node_type(idx_max);

        if(valid_op(idx_max))
            return std::make_shared<ChangeNodeType>(model.name(idx_max), node_type.opposite(), *max_element);
        else
            return nullptr;
    }

    template<typename Model>
    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(Model& model, OperatorTabuSet& tabu_set) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }
        auto delta_ptr = delta.data();
        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            int idx_max = *it;
            auto node_type = model.node_type(idx_max);
            std::shared_ptr<Operator> op = std::make_shared<ChangeNodeType>(model.name(idx_max), node_type.opposite(), delta(idx_max));
            if (tabu_set.contains(op))
                return std::move(op);

        }

        return nullptr;
    }

    template<typename Model>
    void ChangeNodeTypeSet::update_scores(Model& model, Operator& op) {
        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->update_local_score(model, *m_score, op);
        } else if(this->owns_local_cache()) {
            this->m_local_cache->update_local_score(model, *m_score, op);
        }
        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_local_delta(model, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_local_delta(model, dwn_op.source());
                update_local_delta(model, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                int index = model.index(dwn_op.node());
                delta(index) = -dwn_op.delta();
            }
                break;
        }
    }

    class OperatorPool {
    public:
        template<typename Model>
        OperatorPool(Model& model, 
                     std::shared_ptr<Score>& score, 
                     std::vector<std::shared_ptr<OperatorSet>> op_sets) : m_score(score),
                                                                          local_cache(std::make_shared<LocalScoreCache>(model)),
                                                                          m_op_sets(std::move(op_sets)) {
            if (m_op_sets.empty()) {
                throw std::invalid_argument("op_sets argument cannot be empty.");
            }
            
            for (auto& op_set : m_op_sets) {
                op_set->set_local_score_cache(local_cache);
            }
        }
        
        template<typename Model>
        void cache_scores(Model& model);
        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model);
        template<typename Model>
        std::shared_ptr<Operator> find_max(Model& model, OperatorTabuSet& tabu_set);
        template<typename Model>
        void update_scores(Model& model, Operator& op);
               
        double score() {
            return local_cache->sum();
        }

        template<typename Model>
        double score(Model& model) {
            double s = 0;
            for (int i = 0; i < model.num_nodes(); ++i) {
                s += m_score->local_score(model, i);
            }
            return s;
        }

        void set_arc_blacklist(const ArcVector& blacklist) {
            for(auto& opset : m_op_sets) {
                opset->set_arc_blacklist(blacklist);
            }
        }
        void set_arc_whitelist(const ArcVector& whitelist) {
            for(auto& opset : m_op_sets) {
                opset->set_arc_whitelist(whitelist);
            }
        }
        void set_max_indegree(int indegree) {
            for(auto& opset : m_op_sets) {
                opset->set_max_indegree(indegree);
            }
        }
        void set_type_whitelist(const FactorTypeVector& type_whitelist) {
            for(auto& opset : m_op_sets) {
                opset->set_type_whitelist(type_whitelist);
            }
        }
    private:
        std::shared_ptr<Score> m_score;
        std::shared_ptr<LocalScoreCache> local_cache;
        std::vector<std::shared_ptr<OperatorSet>> m_op_sets;
    };

    template<typename Model>
    void OperatorPool::cache_scores(Model& model) {
        local_cache->cache_local_scores(model, *m_score);

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores(model);
        }
    }

    template<typename Model>
    std::shared_ptr<Operator> OperatorPool::find_max(Model& model) {

        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto it = m_op_sets.begin(); it != m_op_sets.end(); ++it) {
            auto new_op = (*it)->find_max(model);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    template<typename Model>
    std::shared_ptr<Operator> OperatorPool::find_max(Model& model, OperatorTabuSet& tabu_set) {
        if (tabu_set.empty())
            return find_max(model);
        
        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto it = m_op_sets.begin(); it != m_op_sets.end(); ++it) {
            auto new_op = (*it)->find_max(model, tabu_set);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    template<typename Model>
    void OperatorPool::update_scores(Model& model, Operator& op) {
        local_cache->update_local_score(model, *m_score, op);
        for (auto& op_set : m_op_sets) {
            op_set->update_scores(model, op);
        }
    }

}

#endif //PGM_DATASET_OPERATORS_HPP