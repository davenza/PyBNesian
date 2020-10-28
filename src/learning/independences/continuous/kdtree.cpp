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

    void KDTree::fit(DataFrame df, int leafsize) {
        m_df = df;
        m_column_names = df.column_names();
        m_datatype = df.same_type();
        m_indices.resize(df->num_rows());
        std::iota(m_indices.begin(), m_indices.end(), 0);
        m_maxes = VectorXd(df->num_columns());
        m_mines = VectorXd(df->num_columns());

        switch (m_datatype) {
                case Type::DOUBLE: {
                    for (int j = 0; j < df->num_columns(); ++j) {
                        m_mines(j) = df.min<arrow::DoubleType>(j);
                        m_maxes(j) = df.max<arrow::DoubleType>(j);
                    }

                    break;
                }
                case Type::FLOAT: {
                    for (int j = 0; j < df->num_columns(); ++j) {
                        m_mines(j) = df.min<arrow::FloatType>(j);
                        m_maxes(j) = df.max<arrow::FloatType>(j);
                    }
                    break;
                }
                default:
                    throw std::invalid_argument("Wrong data type to apply KDTree.");
        }

        m_root = build_kdtree(df, leafsize);
    }
    
    std::vector<std::pair<VectorXd, VectorXi>> KDTree::query(const DataFrame& test_df, int k, double p) const {
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
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (p == 2) {
                    EuclideanDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (std::isinf(p)) {
                    ChebyshevDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else {
                    MinkowskiP<arrow::DoubleType> dist(train_downcast, test_downcast, p);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
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
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (p == 2) {
                    EuclideanDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else if (std::isinf(p)) {
                    ChebyshevDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
                        auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                        res.push_back(t);
                    }
                } else {
                    MinkowskiP<arrow::FloatType> dist(train_downcast, test_downcast, p);
                    for (int i = 0; i < test_df->num_rows(); ++i) {
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

    std::tuple<VectorXi, VectorXi, VectorXi> KDTree::count_conditional_subspaces(
                                const DataFrame& test_df,
                                size_t x,
                                size_t y,
                                const std::vector<size_t>::iterator z_begin,
                                const std::vector<size_t>::iterator z_end,
                                const VectorXd& eps) const {
        VectorXi count_xz(test_df->num_rows());
        VectorXi count_yz(test_df->num_rows());
        VectorXi count_z(test_df->num_rows());

        // std::vector<bool> in_z(m_df->num_columns());
        std::vector<int> z_order(m_df->num_columns());
        std::fill(z_order.begin(), z_order.end(), -1);
            
        auto i = 0;
        for (auto it = z_begin; it != z_end; ++it, ++i) {
            z_order[*it] = i;
        }

        switch(m_datatype) {
            case Type::DOUBLE: {
                auto train_z = m_df.downcast_vector<arrow::DoubleType>(z_begin, z_end);
                auto test_z = test_df.downcast_vector<arrow::DoubleType>(z_begin, z_end);
                ChebyshevDistance<arrow::DoubleType> dist(train_z, test_z);

                auto test_data = test_df.downcast_vector<arrow::DoubleType>();


                for(int i = 0; i < test_df->num_rows(); ++i) {
                    auto c = count_instance_conditional_subspaces<arrow::DoubleType>(test_data,
                                                                                     i,
                                                                                     x,
                                                                                     y,
                                                                                     z_begin,
                                                                                     z_end,
                                                                                     z_order,
                                                                                     dist,
                                                                                     eps(i));
                    
                    count_xz(i) = std::get<0>(c);
                    count_yz(i) = std::get<1>(c);
                    count_z(i) = std::get<2>(c);
                }
                break;
            }
            case Type::FLOAT: {
                auto train_z = m_df.downcast_vector<arrow::FloatType>(z_begin, z_end);
                auto test_z = test_df.downcast_vector<arrow::FloatType>(z_begin, z_end);
                ChebyshevDistance<arrow::FloatType> dist(train_z, test_z);

                auto test_data = test_df.downcast_vector<arrow::FloatType>();

                for(int i = 0; i < test_df->num_rows(); ++i) {
                    auto c = count_instance_conditional_subspaces<arrow::FloatType>(test_data,
                                                                                    i,
                                                                                    x,
                                                                                    y,
                                                                                    z_begin,
                                                                                    z_end,
                                                                                    z_order,
                                                                                    dist,
                                                                                    eps(i));
                    
                    count_xz(i) = std::get<0>(c);
                    count_yz(i) = std::get<1>(c);
                    count_z(i) = std::get<2>(c);
                }
                break;
            }
            default:
                throw std::invalid_argument("Wrong data type to apply KDTree.");

        }

        return std::make_tuple(count_xz, count_yz, count_z);
    }
}

