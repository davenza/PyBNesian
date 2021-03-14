#include <learning/independences/continuous/RCoT.hpp>

namespace learning::independences::continuous {
    
    template<typename ArrowType, typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y) const {
        if (m_df.null_count(x, y) == 0) {
            auto x_vec = m_df.to_eigen<false, ArrowType, false>(x);
            auto y_vec = m_df.to_eigen<false, ArrowType, false>(y);
            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;
            return RIT<false>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
        } else {
            auto combined_bitmap = m_df.combined_bitmap(x, y);
            auto x_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, x);
            auto y_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, y);
            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;
            return RIT<true>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
        }
    }

    template<typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y) const {
        auto type = m_df.same_type(x, y);
        switch (type) {
            case Type::DOUBLE: {
                return pvalue<arrow::DoubleType>(x, y);
            }
            case Type::FLOAT: {
                return pvalue<arrow::FloatType>(x, y);
            }
            default:
                throw std::invalid_argument("Column are not continuous.");
        }
    }

    double RCoT::pvalue(int v1, int v2) const {
        return pvalue<>(v1, v2);
    }

    double RCoT::pvalue(const std::string& v1, const std::string& v2) const {
        return pvalue<>(v1, v2);
    }

    template<typename ArrowType, typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y, const VarType& z) const {
        if (m_df.null_count(x, y, z) == 0) {
            auto x_vec = m_df.to_eigen<false, ArrowType, false>(x);
            auto y_vec = m_df.to_eigen<false, ArrowType, false>(y);
            
            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;

            auto z_vec = m_df.to_eigen<false, ArrowType, false>(z);
            
            if (util::sse(*z_vec) == 0) {
                return RIT<false>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
            } else {
                return RSingleZ<false>(m_df.index(x), m_df.index(y), m_df.index(z), *x_vec, *y_vec, *z_vec);
            }
        } else {
            auto combined_bitmap = m_df.combined_bitmap(x, y, z);
            auto x_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, x);
            auto y_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, y);

            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;

            auto z_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, z);

            if (util::sse(*z_vec) == 0) {
                return RIT<true>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
            } else {
                return RSingleZ<true>(m_df.index(x), m_df.index(y), m_df.index(z), *x_vec, *y_vec, *z_vec);
            }
        }
    }

    template<typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y, const VarType& z) const {
        auto type = m_df.same_type(x, y, z);
        switch (type) {
            case Type::DOUBLE: {
                return pvalue<arrow::DoubleType>(x, y, z);
            }
            case Type::FLOAT: {
                return pvalue<arrow::FloatType>(x, y, z);
            }
            default:
                throw std::invalid_argument("Column are not continuous.");
        }
    }

    double RCoT::pvalue(int v1, int v2, int cond) const {
        return pvalue<>(v1, v2, cond);
    }

    double RCoT::pvalue(const std::string& v1, const std::string& v2, const std::string& cond) const {
        return pvalue<>(v1, v2, cond);
    }

    template<typename ArrowType, typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y, const std::vector<VarType>& z) const {
        if (m_df.null_count(x, y, z) == 0) {
            auto x_vec = m_df.to_eigen<false, ArrowType, false>(x);
            auto y_vec = m_df.to_eigen<false, ArrowType, false>(y);

            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;

            auto z_mat = m_df.to_eigen<false, ArrowType, false>(z);
            
            auto z_sse = util::sse_cols(*z_mat);
            
            bool zall_valid = true;
            for (auto i = 0; i < z_sse.rows(); ++i) {
                if (z_sse(i) == 0) {
                    zall_valid = false;
                    break;
                }
            }

            if (!zall_valid) {
                std::vector<std::string> valid_names;

                for (auto i = 0; i < z_sse.rows(); ++i) {
                    if (z_sse(i) > 0) {
                        valid_names.push_back(m_df.name(z[i]));
                    }
                }

                if (!valid_names.empty())
                    z_mat = m_df.to_eigen<false, ArrowType, false>(valid_names);
                else
                    return RIT<false>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
            }

            return RMultiZ<false>(m_df.index(x), m_df.index(y), *x_vec, *y_vec, *z_mat);
        } else {
            auto combined_bitmap = m_df.combined_bitmap(x, y, z);
            auto x_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, x);
            auto y_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, y);

            if (util::sse(*x_vec) == 0 || util::sse(*y_vec) == 0)
                return 1;

            auto z_mat = m_df.to_eigen<false, ArrowType>(combined_bitmap, z);

            auto z_sse = util::sse_cols(*z_mat);
            
            bool zall_valid = true;
            for (auto i = 0; z_sse.rows(); ++i) {
                if (z_sse(i) == 0) {
                    zall_valid = false;
                    break;
                }
            }

            if (!zall_valid) {
                std::vector<std::string> valid_names;

                for (auto i = 0; z_sse.rows(); ++i) {
                    if (z_sse(i) > 0) {
                        valid_names.push_back(m_df.name(z[i]));
                    }
                }

                if (!valid_names.empty()) {
                    combined_bitmap = m_df.combined_bitmap(x, y, valid_names);
                    x_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, x);
                    y_vec = m_df.to_eigen<false, ArrowType>(combined_bitmap, y);
                    z_mat = m_df.to_eigen<false, ArrowType>(valid_names);
                } else {
                    return RIT<true>(m_df.index(x), m_df.index(y), *x_vec, *y_vec);
                }
            }

            return RMultiZ<true>(m_df.index(x), m_df.index(y), *x_vec, *y_vec, *z_mat);
        }
    }

    template<typename VarType>
    double RCoT::pvalue(const VarType& x, const VarType& y, const std::vector<VarType>& z) const {
        auto type = m_df.same_type(x, y, z);
        switch (type) {
            case Type::DOUBLE: {
                return pvalue<arrow::DoubleType>(x, y, z);
            }
            case Type::FLOAT: {
                return pvalue<arrow::FloatType>(x, y, z);
            }
            default:
                throw std::invalid_argument("Column are not continuous.");
        }
    }

    double RCoT::pvalue(int x, int y, const std::vector<int>& z) const {
        return pvalue<>(x, y, z);
    }

    double RCoT::pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
        return pvalue<>(x, y, z);
    }

}
