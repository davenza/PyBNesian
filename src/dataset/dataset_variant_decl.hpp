public:

    using EigenVectorVariant = std::variant<
                                std::unique_ptr<Matrix<double, Dynamic, 2>>, // append ones
                                std::unique_ptr<Matrix<double, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                std::unique_ptr<Matrix<float, Dynamic, 2>>, // append ones
                                std::unique_ptr<Matrix<float, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                std::unique_ptr<Map<const Matrix<double, Dynamic, 1>>>, // !append_ones and null_bitmap == nullptr
                                std::unique_ptr<Map<const Matrix<float, Dynamic, 1>>> // !append_ones and null_bitmap == nullptr
                            >;

    //        FIXME: Keeping Map may not worth it. Check benchmarks.
    using EigenMatrixVariant = std::variant<
                                std::unique_ptr<MatrixXd>,
                                std::unique_ptr<MatrixXf>,
    //                                        If cols.size() == 1, same variants as the EigenVectorVariant
                                std::unique_ptr<Matrix<double, Dynamic, 2>>, // append ones
                                std::unique_ptr<Matrix<double, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                std::unique_ptr<Matrix<float, Dynamic, 2>>, // append ones
                                std::unique_ptr<Matrix<float, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                std::unique_ptr<Map<const Matrix<double, Dynamic, 1>>>, // !append_ones and null_bitmap == nullptr
                                std::unique_ptr<Map<const Matrix<float, Dynamic, 1>>> // !append_ones and null_bitmap == nullptr
                            >;

    template<bool append_ones, typename T, util::enable_if_index_container_t<T, int> = 0>
    EigenMatrixVariant to_eigen_variant(T cols) const;
    template<bool append_ones, typename T, util::enable_if_index_container_t<T, int> = 0>
    EigenMatrixVariant to_eigen_variant(T cols, Buffer_ptr bitmap) const;
    template<bool append_ones, typename V>
    EigenMatrixVariant to_eigen_variant(std::initializer_list<V> cols) const {
    return to_eigen_variant<append_ones, std::initializer_list<V>>(cols);
    }
    template<bool append_ones, typename V>
    EigenMatrixVariant to_eigen_variant(std::initializer_list<V> cols, Buffer_ptr bitmap) const {
    return to_eigen_variant<append_ones, std::initializer_list<V>>(cols, bitmap);
    }
    template<bool append_ones, int = 0>
    EigenVectorVariant to_eigen_variant(int i) const;
    template<bool append_ones, int = 0>
    EigenVectorVariant to_eigen_variant(int i, Buffer_ptr bitmap) const;
    template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    EigenVectorVariant to_eigen_variant(StringType name) const;
    template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    EigenVectorVariant to_eigen_variant(StringType name, Buffer_ptr bitmap) const;

private:
    template<bool append_ones, typename T, typename ArrowType, util::enable_if_index_container_t<T, int> = 0>
    EigenMatrixVariant to_eigen_variant_typed(T cols, Buffer_ptr bitmap) const;

    template<bool append_ones, typename ArrowType>
    EigenVectorVariant to_eigen_variant_typed(Array_ptr c, Buffer_ptr bitmap) const;