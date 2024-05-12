#ifndef PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
#define PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP

#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <util/vector.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
using VectorXb = Matrix<bool, Dynamic, 1>;

using factors::FactorType;
using learning::scores::Score, learning::scores::ValidatedScore;
using models::BayesianNetwork, models::BayesianNetworkBase;
using models::ConditionalBayesianNetworkBase;
using util::ArcStringVector, util::FactorTypeVector;

namespace learning::operators {

class Operator {
public:
    Operator(double delta) : m_delta(delta) {}
    virtual ~Operator(){};

    virtual bool is_python_derived() const { return false; }

    virtual void apply(BayesianNetworkBase& m) const = 0;
    virtual std::vector<std::string> nodes_changed(const BayesianNetworkBase&) const = 0;
    virtual std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase&) const = 0;
    virtual std::shared_ptr<Operator> opposite(const BayesianNetworkBase&) const = 0;
    virtual std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase&) const = 0;
    double delta() const { return m_delta; }

    virtual std::string ToString() const = 0;

    virtual std::size_t hash() const = 0;
    virtual bool operator==(const Operator& a) const = 0;
    bool operator!=(const Operator& a) const { return !(*this == a); }

    static std::shared_ptr<Operator>& keep_python_alive(std::shared_ptr<Operator>& op) {
        if (op && op->is_python_derived()) {
            auto o = py::cast(op);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<Operator*>();
            op = std::shared_ptr<Operator>(keep_python_state_alive, ptr);
        }

        return op;
    }

    static std::shared_ptr<Operator> keep_python_alive(const std::shared_ptr<Operator>& op) {
        if (op && op->is_python_derived()) {
            auto o = py::cast(op);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<Operator*>();
            return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
        }

        return op;
    }

private:
    double m_delta;
};

template <typename DerivedOperator>
bool equal_operator(const DerivedOperator& tthis, const Operator& op) {
    if (auto d = dynamic_cast<const DerivedOperator*>(&op)) {
        return tthis == (*d);
    } else {
        return false;
    }
}

class ArcOperator : public Operator {
public:
    ArcOperator(std::string source, std::string target, double delta)
        : Operator(delta), m_source(source), m_target(target) {}

    const std::string& source() const { return m_source; }
    const std::string& target() const { return m_target; }

private:
    std::string m_source;
    std::string m_target;
};

class AddArc : public ArcOperator {
public:
    AddArc(std::string source, std::string target, double delta) : ArcOperator(source, target, delta) {}

    void apply(BayesianNetworkBase& m) const override { m.add_arc_unsafe(this->source(), this->target()); }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase&) const override { return {this->target()}; }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase&) const override {
        return {this->target()};
    }

    std::shared_ptr<Operator> opposite(const BayesianNetworkBase&) const override;
    std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase&) const override;

    std::string ToString() const override {
        return "AddArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) + ")";
    }

    std::size_t hash() const override {
        size_t seed = 3;
        util::hash_combine(seed, this->source());
        util::hash_combine(seed, this->target());
        return seed;
    }

    bool operator==(const Operator& op) const override { return equal_operator<AddArc>(*this, op); }

    bool operator==(const AddArc& other) const {
        return this->source() == other.source() && this->target() == other.target();
    }
};

class RemoveArc : public ArcOperator {
public:
    RemoveArc(std::string source, std::string target, double delta) : ArcOperator(source, target, delta) {}

    void apply(BayesianNetworkBase& m) const override { m.remove_arc(this->source(), this->target()); }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase&) const override { return {this->target()}; }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase&) const override {
        return {this->target()};
    }

    std::shared_ptr<Operator> opposite(const BayesianNetworkBase&) const override {
        return std::make_shared<AddArc>(this->source(), this->target(), -this->delta());
    }

    std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase& m) const override {
        return opposite(static_cast<const BayesianNetworkBase&>(m));
    }

    std::string ToString() const override {
        return "RemoveArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) +
               ")";
    }

    std::size_t hash() const override {
        size_t seed = 5;
        util::hash_combine(seed, this->source());
        util::hash_combine(seed, this->target());
        return seed;
    }

    bool operator==(const Operator& op) const override { return equal_operator<RemoveArc>(*this, op); }

    bool operator==(const RemoveArc& other) const {
        return this->source() == other.source() && this->target() == other.target();
    }
};

class FlipArc : public ArcOperator {
public:
    FlipArc(std::string source, std::string target, double delta) : ArcOperator(source, target, delta) {}

    void apply(BayesianNetworkBase& m) const override { m.flip_arc_unsafe(this->source(), this->target()); }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase&) const override {
        return {this->source(), this->target()};
    }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase&) const override {
        return {this->source(), this->target()};
    }

    std::shared_ptr<Operator> opposite(const BayesianNetworkBase&) const override {
        return std::make_shared<FlipArc>(this->target(), this->source(), -this->delta());
    }

    std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase& m) const override {
        return opposite(static_cast<const BayesianNetworkBase&>(m));
    }

    std::string ToString() const override {
        return "FlipArc(" + this->source() + " -> " + this->target() + "; Delta: " + std::to_string(this->delta()) +
               ")";
    }

    std::size_t hash() const override {
        size_t seed = 7;
        util::hash_combine(seed, this->source());
        util::hash_combine(seed, this->target());
        return seed;
    }

    bool operator==(const Operator& op) const override { return equal_operator<FlipArc>(*this, op); }

    bool operator==(const FlipArc& other) const {
        return this->source() == other.source() && this->target() == other.target();
    }
};

class ChangeNodeType : public Operator {
public:
    ChangeNodeType(std::string node, std::shared_ptr<FactorType> new_node_type, double delta)
        : Operator(delta), m_node(node), m_new_node_type(new_node_type) {}

    const std::string& node() const { return m_node; }
    const std::shared_ptr<FactorType>& node_type() const { return m_new_node_type; }
    void apply(BayesianNetworkBase& m) const override { m.set_node_type(m_node, m_new_node_type); }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase&) const override { return {m_node}; }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase&) const override { return {m_node}; }

    std::shared_ptr<Operator> opposite(const BayesianNetworkBase& m) const override {
        return std::make_shared<ChangeNodeType>(m_node, m.node_type(m_node), -this->delta());
    }

    std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase& m) const override {
        return opposite(static_cast<const BayesianNetworkBase&>(m));
    }

    std::string ToString() const override {
        return "ChangeNodeType(" + node() + " -> " + m_new_node_type->ToString() +
               "; Delta: " + std::to_string(this->delta()) + ")";
    }

    std::size_t hash() const override {
        size_t seed = 9;
        util::hash_combine(seed, m_node);
        util::hash_combine(seed, m_new_node_type->hash());
        return seed;
    }

    bool operator==(const Operator& op) const override { return equal_operator<ChangeNodeType>(*this, op); }

    bool operator==(const ChangeNodeType& other) const {
        return m_node == other.m_node && m_new_node_type == other.m_new_node_type;
    }

private:
    std::string m_node;
    std::shared_ptr<FactorType> m_new_node_type;
};

class HashOperator {
public:
    inline std::size_t operator()(const std::shared_ptr<Operator>& op) const { return op->hash(); }
};

class OperatorPtrEqual {
public:
    inline bool operator()(const std::shared_ptr<Operator>& lhs, const std::shared_ptr<Operator>& rhs) const {
        return (*lhs) == (*rhs);
    }
};

class OperatorTabuSet {
public:
    OperatorTabuSet() : m_set() {}

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
    OperatorTabuSet& operator=(OperatorTabuSet&& other) {
        m_set = std::move(other.m_set);
        return *this;
    }

    void insert(const std::shared_ptr<Operator>& op) { m_set.insert(op); }

    bool contains(const std::shared_ptr<Operator>& op) const { return m_set.count(op) > 0; }
    void clear() { m_set.clear(); }
    bool empty() const { return m_set.empty(); }

private:
    using SetType = std::unordered_set<std::shared_ptr<Operator>, HashOperator, OperatorPtrEqual>;

    SetType m_set;
};

class LocalScoreCache {
public:
    LocalScoreCache() : m_local_score() {}
    LocalScoreCache(const BayesianNetworkBase& m) : m_local_score(m.num_nodes()) {}

    void cache_local_scores(const BayesianNetworkBase& model, const Score& score) {
        if (m_local_score.rows() != model.num_nodes()) {
            m_local_score = VectorXd(model.num_nodes());
        }

        for (const auto& node : model.nodes()) {
            m_local_score(model.collapsed_index(node)) = score.local_score(model, node);
        }
    }

    void cache_vlocal_scores(const BayesianNetworkBase& model, const ValidatedScore& score) {
        if (m_local_score.rows() != model.num_nodes()) {
            m_local_score = VectorXd(model.num_nodes());
        }

        for (const auto& node : model.nodes()) {
            m_local_score(model.collapsed_index(node)) = score.vlocal_score(model, node);
        }
    }

    void update_local_score(const BayesianNetworkBase& model, const Score& score, const std::string& variable) {
        m_local_score(model.collapsed_index(variable)) = score.local_score(model, variable);
    }

    void update_vlocal_score(const BayesianNetworkBase& model,
                             const ValidatedScore& score,
                             const std::string& variable) {
        m_local_score(model.collapsed_index(variable)) = score.vlocal_score(model, variable);
    }

    double sum() { return m_local_score.sum(); }

    double local_score(const BayesianNetworkBase& model, const std::string& name) {
        return m_local_score(model.collapsed_index(name));
    }

private:
    VectorXd m_local_score;
};

class OperatorSet {
public:
    OperatorSet() : m_local_cache(nullptr), m_owns_local_cache(false) {}
    virtual ~OperatorSet() {}
    virtual bool is_python_derived() const { return false; }
    virtual void cache_scores(const BayesianNetworkBase&, const Score&) = 0;
    virtual std::shared_ptr<Operator> find_max(const BayesianNetworkBase&) const = 0;
    virtual std::shared_ptr<Operator> find_max(const BayesianNetworkBase&, const OperatorTabuSet&) const = 0;
    virtual void update_scores(const BayesianNetworkBase&, const Score&, const std::vector<std::string>&) = 0;

    virtual void cache_scores(const ConditionalBayesianNetworkBase&, const Score&) = 0;
    virtual std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase&) const = 0;
    virtual std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase&, const OperatorTabuSet&) const = 0;
    virtual void update_scores(const ConditionalBayesianNetworkBase&,
                               const Score&,
                               const std::vector<std::string>&) = 0;

    void set_local_score_cache(std::shared_ptr<LocalScoreCache> score_cache) {
        m_local_cache = score_cache;
        m_owns_local_cache = false;
    }

    std::shared_ptr<LocalScoreCache> local_score_cache() { return m_local_cache; }

    virtual void set_arc_blacklist(const ArcStringVector&) {};
    virtual void set_arc_whitelist(const ArcStringVector&) {};
    virtual void set_max_indegree(int) {};
    virtual void set_type_blacklist(const FactorTypeVector&) {};
    virtual void set_type_whitelist(const FactorTypeVector&) {};
    virtual void finished() { m_local_cache = nullptr; }

    static std::shared_ptr<OperatorSet>& keep_python_alive(std::shared_ptr<OperatorSet>& op_set) {
        if (op_set && op_set->is_python_derived()) {
            auto o = py::cast(op_set);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<OperatorSet*>();
            op_set = std::shared_ptr<OperatorSet>(keep_python_state_alive, ptr);
        }

        return op_set;
    }

    static std::shared_ptr<OperatorSet> keep_python_alive(const std::shared_ptr<OperatorSet>& op_set) {
        if (op_set && op_set->is_python_derived()) {
            auto o = py::cast(op_set);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<OperatorSet*>();
            return std::shared_ptr<OperatorSet>(keep_python_state_alive, ptr);
        }

        return op_set;
    }

    static std::vector<std::shared_ptr<OperatorSet>>& keep_vector_python_alive(
        std::vector<std::shared_ptr<OperatorSet>>& v) {
        for (auto& op_set : v) {
            OperatorSet::keep_python_alive(op_set);
        }

        return v;
    }

    static std::vector<std::shared_ptr<OperatorSet>> keep_vector_python_alive(
        const std::vector<std::shared_ptr<OperatorSet>>& v) {
        std::vector<std::shared_ptr<OperatorSet>> fv;
        fv.reserve(v.size());

        for (const auto& op_set : v) {
            fv.push_back(OperatorSet::keep_python_alive(op_set));
        }

        return fv;
    }

protected:
    bool owns_local_cache() const { return m_owns_local_cache; }

    template <typename M>
    void initialize_local_cache(M& model) {
        if (!this->m_local_cache) {
            m_local_cache = std::make_shared<LocalScoreCache>(model);
            m_owns_local_cache = true;
        }
    }

    void raise_uninitialized() const {
        if (m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }
    }

    std::shared_ptr<LocalScoreCache> m_local_cache;
    bool m_owns_local_cache;
};

class ArcOperatorSet : public OperatorSet {
public:
    ArcOperatorSet(ArcStringVector blacklist = ArcStringVector(),
                   ArcStringVector whitelist = ArcStringVector(),
                   int indegree = 0)
        : delta(), valid_op(), sorted_idx(), m_blacklist(blacklist), m_whitelist(whitelist), max_indegree(indegree) {}

    void cache_scores(const BayesianNetworkBase& model, const Score& score) override;
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override;
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override;
    template <bool limited_indigree>
    std::shared_ptr<Operator> find_max_indegree(const BayesianNetworkBase& model) const;
    template <bool limited_indigree>
    std::shared_ptr<Operator> find_max_indegree(const BayesianNetworkBase& model,
                                                const OperatorTabuSet& tabu_set) const;
    void update_scores(const BayesianNetworkBase&, const Score&, const std::vector<std::string>&) override;

    void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override;
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override;
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override;
    template <bool limited_indigree>
    std::shared_ptr<Operator> find_max_indegree(const ConditionalBayesianNetworkBase& model) const;
    template <bool limited_indigree>
    std::shared_ptr<Operator> find_max_indegree(const ConditionalBayesianNetworkBase& model,
                                                const OperatorTabuSet& tabu_set) const;
    void update_scores(const ConditionalBayesianNetworkBase&, const Score&, const std::vector<std::string>&) override;

    void update_incoming_arcs_scores(const BayesianNetworkBase& model,
                                     const Score& score,
                                     const std::string& target_node);
    void update_incoming_arcs_scores(const ConditionalBayesianNetworkBase& model,
                                     const Score& score,
                                     const std::string& target_node);

    void update_valid_ops(const BayesianNetworkBase& bn);
    void update_valid_ops(const ConditionalBayesianNetworkBase& bn);

    void set_arc_blacklist(const ArcStringVector& blacklist) override { m_blacklist = blacklist; }

    void set_arc_whitelist(const ArcStringVector& whitelist) override { m_whitelist = whitelist; }

    void set_max_indegree(int indegree) override { max_indegree = indegree; }

private:
    MatrixXd delta;
    MatrixXb valid_op;
    mutable std::vector<int> sorted_idx;
    ArcStringVector m_blacklist;
    ArcStringVector m_whitelist;
    int max_indegree;
};

template <bool limited_indegree>
std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const BayesianNetworkBase& model) const {
    auto delta_ptr = delta.data();

    // TODO: Not checking sorted_idx empty
    std::sort(
        sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) { return delta_ptr[i1] > delta_ptr[i2]; });

    for (auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
        auto idx = *it;
        auto source_collapsed = idx % model.num_nodes();
        auto target_collapsed = idx / model.num_nodes();

        const auto& source = model.collapsed_name(source_collapsed);
        const auto& target = model.collapsed_name(target_collapsed);

        if (model.has_arc(source, target)) {
            return std::make_shared<RemoveArc>(source, target, delta(source_collapsed, target_collapsed));
        } else if (model.has_arc(target, source) && model.can_flip_arc(target, source)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            return std::make_shared<FlipArc>(target, source, delta(source_collapsed, target_collapsed));
        } else if (model.can_add_arc(source, target)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            return std::make_shared<AddArc>(source, target, delta(source_collapsed, target_collapsed));
        }
    }

    return nullptr;
}

template <bool limited_indegree>
std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const ConditionalBayesianNetworkBase& model) const {
    auto delta_ptr = delta.data();

    // TODO: Not checking sorted_idx empty
    std::sort(
        sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) { return delta_ptr[i1] > delta_ptr[i2]; });

    for (auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
        auto idx = *it;
        auto source_joint_collapsed = idx % model.num_joint_nodes();
        auto target_collapsed = idx / model.num_joint_nodes();

        const auto& source = model.joint_collapsed_name(source_joint_collapsed);
        const auto& target = model.collapsed_name(target_collapsed);

        auto d = delta(source_joint_collapsed, target_collapsed);
        if (model.has_arc(source, target)) {
            return std::make_shared<RemoveArc>(source, target, d);
        }

        if (model.is_interface(source)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            // If source is interface, the arc has a unique direction, and cannot produce cycles as source cannot have
            // parents.
            if (model.type_ref().can_have_arc(model, source, target))
                return std::make_shared<AddArc>(source, target, d);
        } else {
            if (model.has_arc(target, source) && model.can_flip_arc(target, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(target) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<FlipArc>(target, source, d);
            } else if (model.can_add_arc(source, target)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(target) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_shared<AddArc>(source, target, d);
            }
        }
    }

    return nullptr;
}

template <bool limited_indegree>
std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const BayesianNetworkBase& model,
                                                            const OperatorTabuSet& tabu_set) const {
    auto delta_ptr = delta.data();

    // TODO: Not checking sorted_idx empty
    std::sort(
        sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) { return delta_ptr[i1] > delta_ptr[i2]; });

    for (auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
        auto idx = *it;
        auto source_collapsed = idx % model.num_nodes();
        auto target_collapsed = idx / model.num_nodes();

        const auto& source = model.collapsed_name(source_collapsed);
        const auto& target = model.collapsed_name(target_collapsed);

        if (model.has_arc(source, target)) {
            std::shared_ptr<Operator> op =
                std::make_shared<RemoveArc>(source, target, delta(source_collapsed, target_collapsed));
            if (!tabu_set.contains(op)) return op;
        } else if (model.has_arc(target, source) && model.can_flip_arc(target, source)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            std::shared_ptr<Operator> op =
                std::make_shared<FlipArc>(target, source, delta(source_collapsed, target_collapsed));
            if (!tabu_set.contains(op)) return op;
        } else if (model.can_add_arc(source, target)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            std::shared_ptr<Operator> op =
                std::make_shared<AddArc>(source, target, delta(source_collapsed, target_collapsed));
            if (!tabu_set.contains(op)) return op;
        }
    }

    return nullptr;
}

template <bool limited_indegree>
std::shared_ptr<Operator> ArcOperatorSet::find_max_indegree(const ConditionalBayesianNetworkBase& model,
                                                            const OperatorTabuSet& tabu_set) const {
    auto delta_ptr = delta.data();

    // TODO: Not checking sorted_idx empty
    std::sort(
        sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) { return delta_ptr[i1] > delta_ptr[i2]; });

    for (auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
        auto idx = *it;
        auto source_joint_collapsed = idx % model.num_joint_nodes();
        auto target_collapsed = idx / model.num_joint_nodes();

        const auto& source = model.joint_collapsed_name(source_joint_collapsed);
        const auto& target = model.collapsed_name(target_collapsed);

        auto d = delta(source_joint_collapsed, target_collapsed);

        if (model.has_arc(source, target)) {
            std::shared_ptr<Operator> op = std::make_shared<RemoveArc>(source, target, d);

            if (!tabu_set.contains(op))
                return op;
            else
                continue;
        }

        if (model.is_interface(source)) {
            if constexpr (limited_indegree) {
                if (model.num_parents(target) >= max_indegree) {
                    continue;
                }
            }
            // If source is interface, the arc has a unique direction, and cannot produce cycles as source cannot have
            // parents.
            if (model.type_ref().can_have_arc(model, source, target)) {
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(source, target, d);
                if (!tabu_set.contains(op)) return op;
            }
        } else {
            if (model.has_arc(target, source) && model.can_flip_arc(target, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(target) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<FlipArc>(target, source, d);
                if (!tabu_set.contains(op)) return op;
            } else if (model.can_add_arc(source, target)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(target) >= max_indegree) {
                        continue;
                    }
                }
                std::shared_ptr<Operator> op = std::make_shared<AddArc>(source, target, d);
                if (!tabu_set.contains(op)) return op;
            }
        }
    }

    return nullptr;
}

class ChangeNodeTypeSet : public OperatorSet {
public:
    ChangeNodeTypeSet(FactorTypeVector blacklist = FactorTypeVector(), FactorTypeVector whitelist = FactorTypeVector())
        : delta(), m_is_whitelisted(), m_type_blacklist(), m_type_whitelist(whitelist) {
        for (const auto& bl : blacklist) {
            m_type_blacklist.insert(bl);
        }
    }

    void cache_scores(const BayesianNetworkBase& model, const Score& score) override;
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override;
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override;
    void update_scores(const BayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& variables) override;

    void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override {
        cache_scores(static_cast<const BayesianNetworkBase&>(model), score);
    }
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override {
        return find_max(static_cast<const BayesianNetworkBase&>(model));
    }
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override {
        return find_max(static_cast<const BayesianNetworkBase&>(model), tabu_set);
    }
    void update_scores(const ConditionalBayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& variables) override {
        update_scores(static_cast<const BayesianNetworkBase&>(model), score, variables);
    }

    void update_whitelisted(const BayesianNetworkBase& model) {
        if (m_is_whitelisted.rows() != model.num_nodes()) {
            m_is_whitelisted = VectorXb(model.num_nodes());
        }

        std::fill(m_is_whitelisted.data(), m_is_whitelisted.data() + model.num_nodes(), false);

        for (const auto& wl : m_type_whitelist) {
            auto index = model.collapsed_index(wl.first);
            m_is_whitelisted(index) = true;
        }
    }

    void set_type_blacklist(const FactorTypeVector& blacklist) override {
        m_type_blacklist.clear();
        for (const auto& bl : blacklist) {
            m_type_blacklist.insert(bl);
        }
    }

    void set_type_whitelist(const FactorTypeVector& whitelist) override { m_type_whitelist = whitelist; }

private:
    std::vector<VectorXd> delta;
    VectorXb m_is_whitelisted;
    util::FactorTypeSet m_type_blacklist;
    FactorTypeVector m_type_whitelist;
};

class OperatorPool : public OperatorSet {
public:
    OperatorPool(std::vector<std::shared_ptr<OperatorSet>> op_sets) : m_op_sets(std::move(op_sets)) {
        if (m_op_sets.empty()) {
            throw std::invalid_argument("op_sets argument cannot be empty.");
        }
    }

    void cache_scores(const BayesianNetworkBase& model, const Score& score) override {
        cache_scores<BayesianNetworkBase>(model, score);
    }
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override {
        return find_max<BayesianNetworkBase>(model);
    }
    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override {
        return find_max<BayesianNetworkBase>(model, tabu_set);
    }
    void update_scores(const BayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& variables) override {
        update_scores<>(model, score, variables);
    }

    void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override {
        cache_scores<ConditionalBayesianNetworkBase>(model, score);
    }
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override {
        return find_max<ConditionalBayesianNetworkBase>(model);
    }
    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu_set) const override {
        return find_max<ConditionalBayesianNetworkBase>(model, tabu_set);
    }
    void update_scores(const ConditionalBayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& variables) override {
        update_scores<>(model, score, variables);
    }

    template <typename M>
    void cache_scores(const M& model, const Score& score);
    template <typename M>
    std::shared_ptr<Operator> find_max(const M& model) const;
    template <typename M>
    std::shared_ptr<Operator> find_max(const M& model, const OperatorTabuSet& tabu_set) const;
    template <typename M>
    void update_scores(const M& model, const Score& score, const std::vector<std::string>& variables);

    void set_arc_blacklist(const ArcStringVector& blacklist) override {
        for (auto& opset : m_op_sets) {
            opset->set_arc_blacklist(blacklist);
        }
    }

    void set_arc_whitelist(const ArcStringVector& whitelist) override {
        for (auto& opset : m_op_sets) {
            opset->set_arc_whitelist(whitelist);
        }
    }

    void set_max_indegree(int indegree) override {
        for (auto& opset : m_op_sets) {
            opset->set_max_indegree(indegree);
        }
    }

    void set_type_whitelist(const FactorTypeVector& type_whitelist) override {
        for (auto& opset : m_op_sets) {
            opset->set_type_whitelist(type_whitelist);
        }
    }

    virtual void finished() override {
        for (auto& opset : m_op_sets) {
            opset->finished();
        }

        OperatorSet::finished();
    }

private:
    std::vector<std::shared_ptr<OperatorSet>> m_op_sets;
};

template <typename M>
void OperatorPool::cache_scores(const M& model, const Score& score) {
    if (!this->m_local_cache) {
        initialize_local_cache(model);

        for (auto& op_set : m_op_sets) {
            op_set->set_local_score_cache(this->local_score_cache());
        }
    }

    m_local_cache->cache_local_scores(model, score);

    for (auto& op_set : m_op_sets) {
        op_set->cache_scores(model, score);
    }
}

template <typename M>
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

template <typename M>
std::shared_ptr<Operator> OperatorPool::find_max(const M& model, const OperatorTabuSet& tabu_set) const {
    raise_uninitialized();

    if (tabu_set.empty()) return find_max(model);

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

template <typename M>
void OperatorPool::update_scores(const M& model, const Score& score, const std::vector<std::string>& variables) {
    raise_uninitialized();

    if (owns_local_cache()) {
        for (const auto& n : variables) {
            m_local_cache->update_local_score(model, score, n);
        }
    }

    for (auto& op_set : m_op_sets) {
        op_set->update_scores(model, score, variables);
    }
}

}  // namespace learning::operators

#endif  // PYBNESIAN_LEARNING_OPERATORS_OPERATORS_HPP
