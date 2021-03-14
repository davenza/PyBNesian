#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_MUTUAL_INFORMATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_MUTUAL_INFORMATION_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>
#include <kdtree/kdtree.hpp>

using dataset::DataFrame, dataset::Copy;
using Eigen::MatrixXi;
using Array_ptr = std::shared_ptr<arrow::Array>;
using kdtree::KDTree, kdtree::IndexComparator;

namespace learning::independences::continuous {
    
    template<typename OutputArrowType, typename InputArrowType>
    DataFrame rank_data(const DataFrame& df) {
        using OutputCType = typename OutputArrowType::c_type;
        arrow::NumericBuilder<OutputArrowType> builder;

        std::vector<Array_ptr> columns;
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);

        std::vector<size_t> indices(df->num_rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<OutputCType> ranked_data(df->num_rows());

        for (int j = 0; j < df->num_columns(); ++j) {
            auto dwn = df.downcast<InputArrowType>(j);
            auto raw_values = dwn->raw_values();
            
            IndexComparator comp(raw_values);
            std::sort(indices.begin(), indices.end(), comp);

            for (int i = 0; i < df->num_rows(); ++i) {
                ranked_data[indices[i]] = static_cast<OutputCType>(i);
            }

            RAISE_STATUS_ERROR(builder.AppendValues(ranked_data.begin(), ranked_data.end()));
            Array_ptr out;
            RAISE_STATUS_ERROR(builder.Finish(&out));
            columns.push_back(out);
            builder.Reset();

            auto f = arrow::field(df.name(j), out->type());
            RAISE_STATUS_ERROR(b.AddField(f));
        }
        RAISE_RESULT_ERROR(auto schema, b.Finish())
        
        auto rb = arrow::RecordBatch::Make(schema, df->num_rows(), columns);
        return DataFrame(rb);
    }

    template<typename OutputArrowType>
    DataFrame rank_data(const DataFrame& df) {
        switch (df.same_type()) {
            case Type::DOUBLE:
                return rank_data<OutputArrowType, arrow::DoubleType>(df);
            case Type::FLOAT:
                return rank_data<OutputArrowType, arrow::FloatType>(df);
            default:
                throw std::invalid_argument("Wrong data type in KMutualInformation.");
        }
    }

    std::tuple<VectorXi, VectorXi, VectorXi> bruteforce_eps_neighbors(const DataFrame& df, const VectorXd& eps);

    double mi_pair(const DataFrame& df, int k);
    double mi_triple(const DataFrame& df, int k);
    double mi_general(const DataFrame& df, int k);
    

    class KMutualInformation : public IndependenceTest {
    public:
        KMutualInformation(DataFrame df, int k, int shuffle_neighbors = 5, int samples = 1000) : 
                    KMutualInformation(df, k, std::random_device{}(), shuffle_neighbors, samples) {}

        KMutualInformation(DataFrame df, int k, unsigned int seed, int shuffle_neighbors = 5, int samples = 1000) : 
                                                                            m_df(df),
                                                                            m_ranked_df(rank_data<arrow::FloatType>(df)),
                                                                            m_k(k),
                                                                            m_seed(seed),
                                                                            m_shuffle_neighbors(shuffle_neighbors),
                                                                            m_samples(samples) {}

        int samples() { return m_samples; }
        void set_samples(int s) { m_samples = s; } 
        unsigned int seed() { return m_seed; }
        void set_seed(unsigned int seed) { m_seed = seed; }
        
        template<typename VarType>
        double pvalue(const VarType& x, const VarType& y) const;
        template<typename VarType>
        double pvalue(const VarType& x, const VarType& y, const VarType& z) const;
        template<typename VarType>
        double pvalue(const VarType& x, const VarType& y, const std::vector<VarType>& z) const;

        double pvalue(int x, int y) const override {
            return pvalue<int>(x, y);
        }
        double pvalue(const std::string& x, const std::string& y) const override {
            return pvalue<std::string>(x, y);
        }

        double pvalue(int x, int y, int z) const override {
            return pvalue<int>(x, y, z);
        }
        double pvalue(const std::string& x, const std::string& y, const std::string& z) const override {
            return pvalue<std::string>(x, y, z);
        }

        double pvalue(int x, int y, const std::vector<int>& z) const override {
            return pvalue<int>(x, y, z);
        }

        double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const override {
            return pvalue<std::string>(x, y, z);
        }

        template<typename MICalculator>
        double shuffled_pvalue(double original_mi,
                               const float* original_rank_x,
                               const DataFrame& z_df,
                               DataFrame& shuffled_df, 
                               const MICalculator mi_calculator) const;

        template<typename VarType>
        double mi(const VarType& x, const VarType& y) const;
        template<typename VarType>
        double mi(const VarType& x, const VarType& y, const VarType& z) const;
        template<typename VarType>
        double mi(const VarType& x, const VarType& y, const std::vector<VarType>& z) const;

        int num_variables() const override { return m_df->num_columns(); }

        std::vector<std::string> variable_names() const override {
            return m_df.column_names();
        }
        
        const std::string& name(int i) const override {
            return m_df.name(i);
        }

        bool has_variables(const std::string& name) const override {
            return m_df.has_columns(name);
        }

        bool has_variables(const std::vector<std::string>& cols) const override {
            return m_df.has_columns(cols);
        }
    private:
        DataFrame m_df;
        DataFrame m_ranked_df;
        int m_k;
        unsigned int m_seed;
        int m_shuffle_neighbors;
        int m_samples;
    };

    template<typename VarType>
    double KMutualInformation::mi(const VarType& x, const VarType& y) const {
        auto subset_df = m_ranked_df.loc(x, y);
        return mi_pair(subset_df, m_k);
    }

    template<typename VarType>
    double KMutualInformation::mi(const VarType& x, const VarType& y, const VarType& z) const {
        auto subset_df = m_ranked_df.loc(x, y, z);
        return mi_triple(subset_df, m_k);
    }

    template<typename VarType>
    double KMutualInformation::mi(const VarType& x, const VarType& y, const std::vector<VarType>& z) const {
        auto subset_df = m_ranked_df.loc(x, y, z);
        return mi_general(subset_df, m_k);
    }

    template<typename VarType>
    double KMutualInformation::pvalue(const VarType& x, const VarType& y) const {
        auto value = mi(x, y);

        auto shuffled_df = m_ranked_df.loc(Copy(x), y);

        auto x_begin = shuffled_df.template mutable_data<arrow::FloatType>(0);
        auto x_end = x_begin + shuffled_df->num_rows();
        std::mt19937 rng {m_seed};

        int count_greater = 0;
        for (int i = 0; i < m_samples; ++i) {
            std::shuffle(x_begin, x_end, rng);
            auto shuffled_value = mi_pair(shuffled_df, m_k);

            if (shuffled_value >= value)
                ++count_greater;
        }

        return static_cast<double>(count_greater) / m_samples;
    }

    template<typename CType, typename Random>
    void shuffle_dataframe(const CType* original_x,
                           CType* shuffled_x, 
                            const std::vector<size_t>& order, 
                            std::vector<bool>& used, 
                            MatrixXi& neighbors,
                            Random& rng) {
        for (int i = 0; i < neighbors.cols(); ++i) {
            auto begin = neighbors.data() + i*neighbors.rows();
            auto end = begin + neighbors.rows();
            std::shuffle(begin, end, rng);
        }

        std::uniform_real_distribution<float> tiebreaker(-0.5, 0.5);
        for (int i = 0; i < neighbors.cols(); ++i) {
            auto index = order[i];

            int neighbor_index = 0;
            for (int j = 0; j < neighbors.rows(); ++j) {
                neighbor_index = neighbors(j, index);
                if (!used[neighbor_index]) {
                    break;
                }
            }
            if (used[neighbor_index]) {
                shuffled_x[index] = original_x[neighbor_index] + tiebreaker(rng);
            } else {
                shuffled_x[index] = original_x[neighbor_index];
                used[neighbor_index] = true;
            }
        }

        std::vector<size_t> sorted_indices(neighbors.cols());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        IndexComparator comp(shuffled_x);
        std::sort(sorted_indices.begin(), sorted_indices.end(), comp);

        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            shuffled_x[sorted_indices[i]] = static_cast<float>(i);
        }
    }

    struct MITriple {
        inline double operator()(const DataFrame& df, int k) const {
            return mi_triple(df, k);
        }
    };

    struct MIGeneral {
        inline double operator()(const DataFrame& df, int k) const {
            return mi_general(df, k);
        }
    };

    template<typename MICalculator>
    double KMutualInformation::shuffled_pvalue(double original_mi,
                                               const float* original_rank_x,
                                               const DataFrame& z_df, 
                                               DataFrame& shuffled_df, 
                                               const MICalculator mi_calculator) const {

        std::mt19937 rng {m_seed};
        MatrixXi neighbors(m_shuffle_neighbors, m_df->num_rows());
        
        KDTree z_tree(z_df);
        auto zknn = z_tree.query(z_df, m_shuffle_neighbors, std::numeric_limits<double>::infinity());

        for (size_t i = 0; i < zknn.size(); ++i) {
            auto indices = zknn[i].second;
            for (int k = 0; k < m_shuffle_neighbors; ++k) {
                neighbors(k, i) = indices[k];
            }
        }

        std::vector<size_t> order(m_df->num_rows());
        std::iota(order.begin(), order.end(), 0);

        std::vector<bool> used(m_df->num_rows());

        auto shuffled_x = shuffled_df.template mutable_data<arrow::FloatType>(0);

        int count_greater = 0;

        for (int i = 0; i < m_samples; ++i) {
            std::shuffle(order.begin(), order.end(), rng);
            shuffle_dataframe(original_rank_x, shuffled_x, order, used, neighbors, rng);

            auto shuffled_value = mi_calculator(shuffled_df, m_k);

            if (shuffled_value >= original_mi)
                ++count_greater;
            
            std::fill(used.begin(), used.end(), false);
        }

        return static_cast<double>(count_greater) / m_samples;
    }


    template<typename VarType>
    double KMutualInformation::pvalue(const VarType& x, const VarType& y, const VarType& z) const {
        auto original_mi = mi(x, y, z);
        auto z_df = m_df.loc(z);
        auto shuffled_df = m_ranked_df.loc(Copy(x), y, z);
        auto original_rank_x = m_ranked_df.template data<arrow::FloatType>(x);

        return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MITriple{});
    }

    template<typename VarType>
    double KMutualInformation::pvalue(const VarType& x, const VarType& y, const std::vector<VarType>& z) const {
        auto original_mi = mi(x, y, z);
        auto z_df = m_df.loc(z);
        auto shuffled_df = m_ranked_df.loc(Copy(x), y, z);
        auto original_rank_x = m_ranked_df.template data<arrow::FloatType>(x);
        
        return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MIGeneral{});
    }

    using DynamicKMutualInformation = DynamicIndependenceTestAdaptator<KMutualInformation>;
}

#endif //PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_MUTUAL_INFORMATION_HPP