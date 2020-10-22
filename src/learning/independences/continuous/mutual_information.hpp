#ifndef PGM_DATASET_MUTUAL_INFORMATION_HPP
#define PGM_DATASET_MUTUAL_INFORMATION_HPP

#include <dataset/dataset.hpp>
#include <learning/independences/continuous/kdtree.hpp>
#include <util/validate_dtype.hpp>
#include <boost/math/special_functions/digamma.hpp>

using dataset::DataFrame, dataset::CopyLOC, dataset::IndexLOC, dataset::Copy;

using Array_ptr = std::shared_ptr<arrow::Array>;
using util::to_type;

namespace learning::independences {
    
    template<typename OutputArrowType, typename InputArrowType>
    DataFrame rank_data(const DataFrame& df) {
        using OutputCType = typename OutputArrowType::c_type;
        arrow::NumericBuilder<OutputArrowType> builder;

        std::vector<Array_ptr> columns;
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);

        std::vector<size_t> indices(df->num_rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<OutputCType> ranked_data(df->num_rows());

        for (size_t j = 0; j < df->num_columns(); ++j) {
            auto dwn = df.downcast<InputArrowType>(j);
            auto raw_values = dwn->raw_values();
            
            IndexComparator comp(raw_values);
            std::sort(indices.begin(), indices.end(), comp);

            for (size_t i = 0; i < df->num_rows(); ++i) {
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

    class KMutualInformation {
    public:
        KMutualInformation(DataFrame df, int k) : m_df(rank_data<arrow::FloatType>(df)),
                                                                            m_k(k),
                                                                            m_indices() {
            for (size_t i = 0; i < m_df->num_columns(); ++i) {
                m_indices.insert(std::make_pair(m_df->column_name(i), i));
            }
        }

        // double mi(int v1, int v2) const;
        // double mi(const std::string& v1, const std::string& v2);
        // double mi(int v1, int v2, int cond) const;
        // double mi(const std::string& v1, const std::string& v2, const std::string& cond) const;
        // double mi(int v1, int v2,
        //             const typename std::vector<int>::const_iterator evidence_begin, 
        //             const typename std::vector<int>::const_iterator evidence_end) const;
        // double mi(const std::string& v1, const std::string& v2,
        //             const typename std::vector<std::string>::const_iterator evidence_begin, 
        //             const typename std::vector<std::string>::const_iterator evidence_end) const;
        template<typename VarType>
        double pvalue(const VarType& v1, const VarType& v2, int samples = 1000) const;

        template<typename VarType>
        double mi(const VarType& v1, const VarType& v2) const;
        template<typename VarType>
        double mi(const VarType& v1, const VarType& v2, const VarType& cond) const;
        template<typename VarType, typename Iter>
        double mi(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const;
    private:
        size_t cached_index(const std::string& name) const { return m_indices.at(name); }
        size_t cached_index(size_t idx) const { return idx; }

        DataFrame m_df;
        int m_k;
        std::unordered_map<std::string, int> m_indices;
    };

    template<typename VarType>
    double KMutualInformation::mi(const VarType& v1, const VarType& v2) const {
        auto cached_v1 = cached_index(v1);
        auto cached_v2 = cached_index(v2);

        auto subset_df = m_df.loc(cached_v1, cached_v2);

        KDTree kdtree(subset_df);
        auto knn_results = kdtree.query(subset_df, m_k+1, std::numeric_limits<float>::infinity());
        
        VectorXd eps(subset_df->num_rows());
        for (auto i = 0; i < subset_df->num_rows(); ++i) {
            eps(i) = knn_results[i].first(m_k);
        }

        auto nv1 = kdtree.count_subspace_eps(subset_df, 0, eps);
        auto nv2 = kdtree.count_subspace_eps(subset_df, 1, eps);

        double res = 0;
        for (int i = 0; i < m_df->num_rows(); ++i) {
            res -= boost::math::digamma(nv1(i)) + boost::math::digamma(nv2(i));
        }
        
        res /= m_df->num_rows();
        res += boost::math::digamma(m_k) + boost::math::digamma(m_df->num_rows());

        return res;
    }

    template<typename VarType>
    double KMutualInformation::mi(const VarType& v1, const VarType& v2, const VarType& cond) const {
        auto cached_x = cached_index(v1);
        auto cached_y = cached_index(v2);
        auto cached_z = cached_index(cond);

        auto subset_df = m_df.loc(cached_x, cached_y, cached_z);

        KDTree kdtree(subset_df);
        auto knn_results = kdtree.query(subset_df, m_k+1, std::numeric_limits<float>::infinity());
        
        VectorXd eps(subset_df->num_rows());
        for (auto i = 0; i < subset_df->num_rows(); ++i) {
            eps(i) = knn_results[i].first(m_k);
        }

        auto [n_xz, n_yz, n_z] = kdtree.count_conditional_subspaces(subset_df, 0, 1, {2}, eps);

        double res = 0;
        for (int i = 0; i < m_df->num_rows(); ++i) {
            res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
        }
        
        res /= m_df->num_rows();
        res += boost::math::digamma(m_k);

        return res;
    }

    template<typename VarType, typename Iter>
    double KMutualInformation::mi(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const {
        auto cached_x = cached_index(v1);
        auto cached_y = cached_index(v2);

        auto subset_df = m_df.loc(cached_x, cached_y, std::make_pair(evidence_begin, evidence_end));

        KDTree kdtree(subset_df);
        auto knn_results = kdtree.query(subset_df, m_k+1, std::numeric_limits<float>::infinity());
        
        VectorXd eps(subset_df->num_rows());
        for (auto i = 0; i < subset_df->num_rows(); ++i) {
            eps(i) = knn_results[i].first(m_k);
        }


        std::vector<size_t> indices(std::distance(evidence_begin, evidence_end));
        std::iota(indices.begin(), indices.end(), 2);
        auto [n_xz, n_yz, n_z] = kdtree.count_conditional_subspaces(subset_df, 0, 1, indices, eps);

        double res = 0;
        for (int i = 0; i < m_df->num_rows(); ++i) {
            res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
        }
        
        res /= m_df->num_rows();
        res += boost::math::digamma(m_k);

        return res;
    }

    template<typename VarType>
    double KMutualInformation::pvalue(const VarType& v1, const VarType& v2, int samples) const {
        auto value = mi(v1, v2);

        auto shuffled_df = m_df.loc(Copy(v1), v2);

        return 0;
    }



}

#endif //PGM_DATASET_MUTUAL_INFORMATION_HPP