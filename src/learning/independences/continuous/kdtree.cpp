#include <learning/independences/continuous/kdtree.hpp>


namespace learning::independences {


    std::unique_ptr<KDTreeNode> KDTree::build_kdtree(const DataFrame& df, int leafsize) {
        switch(df.same_type()) {
            case Type::DOUBLE: {
                return learning::independences::build_kdtree<arrow::DoubleType>(df, leafsize, 
                                                       m_indices.begin(), m_indices.end(), -1, true,
                                                       m_maxes, m_mines);
            }
            case Type::FLOAT: {
                return learning::independences::build_kdtree<arrow::FloatType>(df, leafsize, 
                                                      m_indices.begin(), m_indices.end(), -1, true,
                                                      m_maxes.template cast<float>(), 
                                                      m_mines.template cast<float>());
            }
            default:
                throw std::invalid_argument("Wrong data type to apply KDTree.");
        }
    }
    
    std::vector<std::pair<VectorXd, VectorXi>> KDTree::query(const DataFrame& test_df, int k, double p) {
        if (k >= m_df->num_rows()) {
            throw std::invalid_argument("\"k\" value equal or greater to training data size.");
        }

        test_df.has_columns(m_column_names);

        if (test_df.same_type(m_column_names) != m_datatype) {
            throw std::invalid_argument("Test data type is different from training data types.");
        }

        std::vector<std::pair<VectorXd, VectorXi>> res;
        res.reserve(test_df->num_rows());

        switch (m_datatype) {
            case Type::DOUBLE: {
                auto train_downcast = m_df.downcast_vector<arrow::DoubleType>(m_column_names);
                auto test_downcast = test_df.downcast_vector<arrow::DoubleType>(m_column_names);

                if (p == 1) {
                    ManhattanDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (p == 2) {
                    EuclideanDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (std::isinf(p)) {
                    ChebyshevDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else {
                    MinkowskiP<arrow::DoubleType> dist(train_downcast, test_downcast, p);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                }
                break;
            }
            case Type::FLOAT: {
                auto train_downcast = m_df.downcast_vector<arrow::FloatType>(m_column_names);
                auto test_downcast = test_df.downcast_vector<arrow::FloatType>(m_column_names);

                if (p == 1) {
                    ManhattanDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (p == 2) {
                    EuclideanDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (std::isinf(p)) {
                    ChebyshevDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else {
                    MinkowskiP<arrow::FloatType> dist(train_downcast, test_downcast, p);
                    for (size_t i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                }
                break;
            }
            default:
                throw std::invalid_argument("Wrong data type to apply KDTree.");
        }

        return res;
    }
}

