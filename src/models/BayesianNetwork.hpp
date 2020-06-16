#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag, graph::AdjListDag;
using boost::source;

using graph::arc_vector; 

using factors::continuous::LinearGaussianCPD;
using factors::continuous::SemiparametricCPD;

namespace models {



    // template<typename it> node_iterator(it b, it e) -> node_iterator<typename std::iterator_traits<Iterator>::value_type>;

    template<typename DagType = AdjMatrixDag>
    class BayesianNetwork {
    public:
        using node_descriptor = typename DagType::node_descriptor;
        using edge_descriptor = typename DagType::edge_descriptor;

        using node_iterator_t = typename DagType::node_iterator_t;

        BayesianNetwork(const std::vector<std::string>& nodes);
        BayesianNetwork(const std::vector<std::string>& nodes, const arc_vector& arcs);

        // static void requires(const DataFrame& df);

        int num_nodes() const {
            return g.num_nodes();
        }

        int num_edges() const {
            return g.num_edges();
        }

        const std::vector<std::string>& nodes() const {
            return m_nodes;
        }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        node_descriptor node(int node_index) const {
            return g.node(node_index);
        }

        bool contains_node(const std::string& name) {
            return indices.find(name) != indices.end();
        }

        node_descriptor node(const std::string& node) const {
            return g.node(m_indices.at(node));
        }

        const std::string& name(int node_index) const {
            return m_nodes[node_index];
        }

        const std::string& name(node_descriptor node) const {
            return name(g.index(node));
        }

        int num_parents(node_descriptor node) const {
            return g.num_parents(node);
        }

        int num_parents(int node_index) const {
            return num_parents(g.node(node_index));
        }

        int num_parents(const std::string& node) const {
            return num_parents(m_indices.at(node));
        }

        int num_children(node_descriptor node) const {
            return g.num_children(node);
        }

        int num_children(int node_index) const {
            return num_children(g.node(node_index));
        }

        int num_children(const std::string& node) const {
            return num_children(m_indices.at(node));
        }

        int index(node_descriptor n) const {
            return g.index(n);
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(node_descriptor node) const {
            std::vector<std::reference_wrapper<const std::string>> parents;
            auto it_parents = g.get_parent_edges(node);

            for (auto it = it_parents.first; it != it_parents.second; ++it) {
                auto parent = g.source(*it);
                auto parent_index = g.index(parent);
                parents.push_back(m_nodes[parent_index]);
            }

            return parents;
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(int node_index) const {
            return get_parents(g.node(node_index));
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(const std::string& node) const {
            return get_parents(m_indices.at(node));
        }

        std::vector<int> get_parent_indices(node_descriptor node) const {
            std::vector<int> parent_indices;
            auto it_parents = g.get_parent_edges(node);

            for (auto it = it_parents.first; it != it_parents.second; ++it) {
                parent_indices.push_back(g.index(g.source(*it)));
            }

            return parent_indices;
        }

        std::vector<int> get_parent_indices(int node_index) const {
            return get_parent_indices(g.node(node_index));
        }

        std::vector<int> get_parent_indices(const std::string& node) const {
            return get_parent_indices(m_indices.at(node));
        }

        bool has_edge(node_descriptor source, node_descriptor dest) const {
            return g.has_edge(source, dest);
        }

        bool has_edge(int source, int dest) const {
            return has_edge(g.node(source), g.node(dest));
        }

        bool has_edge(const std::string& source, const std::string& dest) const {
            return has_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool has_path(node_descriptor source, node_descriptor dest) const {
            return g.has_path(source, dest);
        }
        
        bool has_path(int source_index, int dest_index) const {
            return has_path(g.node(source_index), g.node(dest_index));
        }
        
        bool has_path(const std::string& source, const std::string& dest) const {
            return has_path(m_indices.at(source), m_indices.at(dest));
        }

        void add_edge(node_descriptor source, node_descriptor dest) {
            g.add_edge(source, dest);
        }

        void add_edge(int source, int dest) {
            add_edge(g.node(source), g.node(dest), g);
        }

        void add_edge(const std::string& source, const std::string& dest) const {
            add_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool can_add_edge(node_descriptor source, node_descriptor dest) const {
            if (num_parents(source) == 0 || num_children(dest) == 0 || !has_path(dest, source)) {
                return true;
            }

            return false;
        }

        bool can_add_edge(int source_index, int dest_index) const {
            return can_add_edge(node(source_index), node(dest_index));
        }

        bool can_add_edge(const std::string& source, const std::string& dest) const {
            return can_add_edge(m_indices.at(source), m_indices.at(dest));
        }

        bool can_flip_edge(node_descriptor source, node_descriptor dest) {
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

        bool can_flip_edge(int source_index, int dest_index) {
            return can_flip_edge(node(source_index), node(dest_index));
        }

        bool can_flip_edge(const std::string& source, const std::string& dest) {
            return can_flip_edge(m_indices.at(source), m_indices.at(dest));
        }

        void remove_edge(node_descriptor source, node_descriptor dest) {
            g.remove_edge(source, dest);
        }

        void remove_edge(int source, int dest) {
            remove_edge(g.node(source), g.node(dest), g);
        }

        void remove_edge(const std::string& source, const std::string& dest) const {
            remove_edge(m_indices.at(source), m_indices.at(dest));
        }

        void fit(const DataFrame& df) = 0;
        VectorXd logpdf(const DataFrame& df) = 0;
        double slogpdf(const DataFrame& df) = 0;

        void print() const {
            std::cout << "Bayesian network: " << std::endl; 
            for(auto [eit, eend] = g.edges(); eit != eend; ++eit)
                std::cout << name(g.source(*eit)) << " -> " << name(g.target(*eit)) << std::endl;
        }

    private:
        DagType g;
        std::vector<std::string> m_nodes;
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;
    };

    template<typename DagType>
    BayesianNetwork<DagType>::BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size()) {
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }
    };

    template<typename DagType>
    BayesianNetwork<DagType>::BayesianNetwork(const std::vector<std::string>& nodes, 
                                                 const arc_vector& edges) 
                                                 : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size())
    {
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }

        for(auto edge : edges) {
            g.add_edge(node(edge.first), node(edge.second));
        }
    };


    template<typename DagType = AdjMatrixDag>
    class GaussianNetwork : public BayesianNetwork<DagType> {
    public:
        GaussianNetwork(const std::vector<std::string>& nodes) : BayesianNetwork<DagType>(nodes), m_cpds() {}
        GaussianNetwork(const std::vector<std::string>& nodes, const arc_vector& arcs) : BayesianNetwork<DagType>(nodes, arcs), m_cpds() {}

        void add_cpds(const std::vector<LinearGaussianCPD> cpds) {
            
            for (auto& cpd : cpds) {
                if (!this->contains_node(cpd.variable())) {
                    throw std::invalid_argument("CPD ");
                }
            }
        } 
    private:
        std::vector<LinearGaussianCPD> m_cpds;
    };


    template<typename DagType = AdjMatrixDag>
    class SemiparametricBN : public BayesianNetwork<DagType> {
    public:
        SemiparametricBN(const std::vector<std::string>& nodes) : BayesianNetwork<DagType>(nodes), m_cpds() {}
        SemiparametricBN(const std::vector<std::string>& nodes, const arc_vector& arcs) : BayesianNetwork<DagType>(nodes, arcs), m_cpds() {}

        void add_cpds(const std::vector<SemiparametricCPD> cpds) {

        }
    private:
        std::vector<SemiparametricCPD> m_cpds;
    };

    // template<typename DagType = AdjMatrixDag>
    // using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK, DagType>;


    // template<typename DagType = AdjMatrixDag>
    // using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK, DagType>;

    // using GaussianNetwork_M = GaussianNetwork<AdjMatrixDag>;
    // using GaussianNetwork_L = GaussianNetwork<AdjListDag>;

    // template<typename DagType>
    // void GaussianNetwork<DagType>::requires(const DataFrame& df) {
    //     auto schema = df->schema();

    //     if (schema->num_fields() == 0) {
    //         throw std::invalid_argument("Provided dataset does not contain columns.");
    //     }

    //     auto dtid = schema->field(0)->type()->id();

    //     if (dtid != Type::DOUBLE && dtid != Type::FLOAT) {
    //         throw std::invalid_argument("Continuous data (double or float) is needed to learn Gaussian networks. "
    //                                     "Column \"" + schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() + ").");
    //     }

    //     for (auto i = 1; i < schema->num_fields(); ++i) {
    //         auto new_dtid = schema->field(i)->type()->id();
    //         if (dtid != new_dtid)
    //             throw std::invalid_argument("All the columns should have the same data type. "
    //                                         "Column \"" + schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() + "). "
    //                                         "Column \"" + schema->field(i)->name() + "\" (DataType: " + schema->field(i)->type()->ToString() + ").");
    //     }
    // }

}

#endif //PGM_DATASET_BAYESIANNETWORK_HPP