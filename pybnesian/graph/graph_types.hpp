#ifndef PYBNESIAN_GRAPH_GRAPH_TYPES_HPP
#define PYBNESIAN_GRAPH_GRAPH_TYPES_HPP

#include <string>
#include <unordered_set>
#include <util/hash_utils.hpp>

namespace graph {

class PDNode;

class DNode {
public:
    DNode(int idx, std::string name, std::unordered_set<int> parents = {}, std::unordered_set<int> children = {})
        : m_idx(idx), m_name(name), m_parents(parents), m_children(children) {}
    friend class PDNode;

    int index() const { return m_idx; }

    const std::string& name() const { return m_name; }

    const std::unordered_set<int>& parents() const { return m_parents; }

    const std::unordered_set<int>& children() const { return m_children; }

    void add_parent(int p) { m_parents.insert(p); }

    void add_children(int ch) { m_children.insert(ch); }

    void remove_parent(int p) { m_parents.erase(p); }

    void remove_children(int ch) { m_children.erase(ch); }

    bool is_root() const { return m_parents.empty(); }

    bool is_leaf() const { return m_children.empty(); }

    void invalidate() {
        m_idx = -1;
        m_name.clear();
        m_parents.clear();
        m_children.clear();
    }

    bool is_valid() const { return m_idx != -1; }

private:
    int m_idx;
    std::string m_name;
    std::unordered_set<int> m_parents;
    std::unordered_set<int> m_children;
};

using Arc = std::pair<int, int>;

struct ArcHash {
    std::size_t operator()(Arc const& arc) const {
        size_t seed = 1;

        util::hash_combine(seed, arc.first);
        util::hash_combine(seed, arc.second);

        return seed;
    }
};

class UNode {
public:
    UNode(int idx, std::string name, std::unordered_set<int> neighbors = {})
        : m_idx(idx), m_name(name), m_neighbors(neighbors) {}

    friend class PDNode;

    int index() const { return m_idx; }

    const std::string& name() const { return m_name; }

    const std::unordered_set<int>& neighbors() const { return m_neighbors; }

    void add_neighbor(int p) { m_neighbors.insert(p); }

    void remove_neighbor(int p) { m_neighbors.erase(p); }

    void invalidate() {
        m_idx = -1;
        m_name.clear();
        m_neighbors.clear();
    }

    bool is_valid() const { return m_idx != -1; }

private:
    int m_idx;
    std::string m_name;
    std::unordered_set<int> m_neighbors;
};

using Edge = std::pair<int, int>;

// From https://stackoverflow.com/questions/28367913/how-to-stdhash-an-unordered-stdpair
struct EdgeHash {
    std::size_t operator()(Edge const& edge) const {
        size_t seed = 1;

        if (edge.first <= edge.second) {
            util::hash_combine(seed, edge.first);
            util::hash_combine(seed, edge.second);
        } else {
            util::hash_combine(seed, edge.second);
            util::hash_combine(seed, edge.first);
        }

        return seed;
    }
};

struct EdgeEqualTo {
    bool operator()(const Edge& lhs, const Edge& rhs) const {
        return (lhs == rhs) || (lhs.first == rhs.second && lhs.second == rhs.first);
    }
};

class PDNode {
public:
    PDNode(int idx,
           std::string name,
           std::unordered_set<int> parents = {},
           std::unordered_set<int> children = {},
           std::unordered_set<int> neighbors = {})
        : m_idx(idx), m_name(name), m_neighbors(neighbors), m_parents(parents), m_children(children) {}

    PDNode(DNode&& dn)
        : m_idx(dn.m_idx),
          m_name(std::move(dn.m_name)),
          m_neighbors(),
          m_parents(std::move(dn.m_parents)),
          m_children(std::move(dn.m_children)) {}

    PDNode(UNode&& un)
        : m_idx(un.m_idx),
          m_name(std::move(un.m_name)),
          m_neighbors(std::move(un.m_neighbors)),
          m_parents(),
          m_children() {}

    int index() const { return m_idx; }

    const std::string& name() const { return m_name; }

    const std::unordered_set<int>& neighbors() const { return m_neighbors; }

    const std::unordered_set<int>& parents() const { return m_parents; }

    const std::unordered_set<int>& children() const { return m_children; }

    void add_neighbor(int p) { m_neighbors.insert(p); }

    void add_parent(int p) { m_parents.insert(p); }

    void add_children(int ch) { m_children.insert(ch); }

    void remove_neighbor(int p) { m_neighbors.erase(p); }

    void remove_parent(int p) { m_parents.erase(p); }

    void remove_children(int ch) { m_children.erase(ch); }

    bool is_root() const { return m_parents.empty(); }

    bool is_leaf() const { return m_children.empty(); }

    void invalidate() {
        m_idx = -1;
        m_name.clear();
        m_neighbors.clear();
        m_parents.clear();
        m_children.clear();
    }

    bool is_valid() const { return m_idx != -1; }

private:
    int m_idx;
    std::string m_name;
    std::unordered_set<int> m_neighbors;
    std::unordered_set<int> m_parents;
    std::unordered_set<int> m_children;
};

}  // namespace graph

#endif  // PYBNESIAN_GRAPH_GRAPH_TYPES_HPP