#ifndef PYBNESIAN_KDE_PRODUCTKDE_HPP
#define PYBNESIAN_KDE_PRODUCTKDE_HPP

#include <util/pickle.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <opencl/opencl_config.hpp>
#include <util/math_constants.hpp>

using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace kde {

class ProductKDE {
public:
    ProductKDE()
        : m_variables(),
          m_fitted(),
          m_bselector(std::make_shared<NormalReferenceRule>()),
          N(0),
          m_training_type(arrow::float64()) {}

    ProductKDE(std::vector<std::string> variables) : ProductKDE(variables, std::make_shared<NormalReferenceRule>()) {}

    ProductKDE(std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> b_selector)
        : m_variables(variables), m_fitted(false), m_bselector(b_selector), N(0), m_training_type(arrow::float64()) {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        if (m_variables.empty()) {
            throw std::invalid_argument("Cannot create a ProductKDE model with 0 variables");
        }
    }

    const std::vector<std::string>& variables() const { return m_variables; }

    void fit(const DataFrame& df);

    const VectorXd& bandwidth() const { return m_bandwidth; }
    void setBandwidth(VectorXd& new_bandwidth) {
        if (static_cast<size_t>(new_bandwidth.rows()) != m_variables.size())
            throw std::invalid_argument(
                "The bandwidth matrix must be a vector with shape "
                "(" +
                std::to_string(m_variables.size()) + ")");

        m_bandwidth = new_bandwidth;
        if (m_bandwidth.rows() > 0) copy_bandwidth_opencl();
    }

    DataFrame training_data() const;

    int num_instances() const {
        check_fitted();
        return N;
    }
    int num_variables() const { return m_variables.size(); }
    bool fitted() const { return m_fitted; }

    std::shared_ptr<arrow::DataType> data_type() const {
        check_fitted();
        return m_training_type;
    }

    std::shared_ptr<BandwidthSelector> bandwidth_type() const { return m_bselector; }

    VectorXd logl(const DataFrame& df) const;

    template <typename ArrowType>
    cl::Buffer logl_buffer(const DataFrame& df) const;

    double slogl(const DataFrame& df) const;

    void save(const std::string name) { util::save_object(*this, name); }

    py::tuple __getstate__() const;
    static ProductKDE __setstate__(py::tuple& t);
    static ProductKDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("ProductKDE factor not fitted.");
    }

    template <typename ArrowType>
    DataFrame _training_data() const;

    template <typename ArrowType, bool contains_null>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;
    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType>
    void product_logl_mat(cl::Buffer& test_buffer,
                          const unsigned int test_offset,
                          const unsigned int test_length,
                          cl::Buffer& output_mat) const;

    template <typename ArrowType>
    cl::Buffer _logl_impl(cl::Buffer& test_buffer, int m) const;

    void copy_bandwidth_opencl();

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    VectorXd m_bandwidth;
    std::vector<cl::Buffer> m_cl_bandwidth;
    std::vector<cl::Buffer> m_training;
    double m_lognorm_const;
    size_t N;
    std::shared_ptr<arrow::DataType> m_training_type;
};

template <typename ArrowType>
DataFrame ProductKDE::_training_data() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    arrow::NumericBuilder<ArrowType> builder;

    auto& opencl = OpenCLConfig::get();
    VectorType tmp_buffer(N);

    std::vector<Array_ptr> columns;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    for (size_t i = 0; i < m_variables.size(); ++i) {
        opencl.read_from_buffer(tmp_buffer.data(), m_training[i], N);

        auto status = builder.Resize(N);
        RAISE_STATUS_ERROR(builder.AppendValues(tmp_buffer.data(), N));

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        columns.push_back(out);
        builder.Reset();

        auto f = arrow::field(m_variables[i], out->type());
        RAISE_STATUS_ERROR(b.AddField(f));
    }

    RAISE_RESULT_ERROR(auto schema, b.Finish())

    auto rb = arrow::RecordBatch::Make(schema, N, columns);
    return DataFrame(rb);
}

template <typename ArrowType, bool contains_null>
void ProductKDE::_fit(const DataFrame& df) {
    using CType = typename ArrowType::c_type;

    if (static_cast<size_t>(m_bandwidth.rows()) != m_variables.size()) m_bandwidth = VectorXd(m_variables.size());
    m_cl_bandwidth.clear();
    m_training.clear();

    Buffer_ptr combined_bitmap;
    if constexpr (contains_null) combined_bitmap = df.combined_bitmap(m_variables);

    N = df.valid_rows(m_variables);

    auto& opencl = OpenCLConfig::get();

    m_bandwidth = m_bselector->diag_bandwidth(df, m_variables);

    for (size_t i = 0; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>) {
            auto sqrt = std::sqrt(m_bandwidth(i));
            m_cl_bandwidth.push_back(opencl.copy_to_buffer(&sqrt, 1));
        } else {
            auto casted = std::sqrt(static_cast<CType>(m_bandwidth(i)));
            m_cl_bandwidth.push_back(opencl.copy_to_buffer(&casted, 1));
        }

        if constexpr (contains_null) {
            auto column = df.to_eigen<false, ArrowType>(combined_bitmap, m_variables[i]);
            m_training.push_back(opencl.copy_to_buffer(column->data(), N));
        } else {
            auto column = df.to_eigen<false, ArrowType, false>(m_variables[i]);
            m_training.push_back(opencl.copy_to_buffer(column->data(), N));
        }
    }

    m_lognorm_const = -0.5 * static_cast<double>(m_variables.size()) * std::log(2 * util::pi<double>) -
                      0.5 * m_bandwidth.array().log().sum() - std::log(N);
}

template <typename ArrowType>
VectorXd ProductKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_buff = logl_buffer<ArrowType>(df);
    auto& opencl = OpenCLConfig::get();
    if (df.null_count(m_variables) == 0) {
        VectorType read_data(df->num_rows());
        opencl.read_from_buffer(read_data.data(), logl_buff, df->num_rows());
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto m = df.valid_rows(m_variables);
        VectorType read_data(m);
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();

        opencl.read_from_buffer(read_data.data(), logl_buff, m);

        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    }
}

template <typename ArrowType>
cl::Buffer ProductKDE::logl_buffer(const DataFrame& df) const {
    auto& opencl = OpenCLConfig::get();

    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();
    // std::ve
    auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m * m_variables.size());

    return _logl_impl<ArrowType>(test_buffer, m);
}

template <typename ArrowType>
void ProductKDE::product_logl_mat(cl::Buffer& test_buffer,
                                  const unsigned int test_offset,
                                  const unsigned int test_length,
                                  cl::Buffer& output_mat) const {
    using CType = typename ArrowType::c_type;

    auto& opencl = OpenCLConfig::get();
    auto& k_logl_values_1d_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logl_values_1d_mat);
    k_logl_values_1d_mat.setArg(0, m_training[0]);
    k_logl_values_1d_mat.setArg(1, static_cast<unsigned int>(N));
    k_logl_values_1d_mat.setArg(2, test_buffer);
    k_logl_values_1d_mat.setArg(3, test_offset);
    k_logl_values_1d_mat.setArg(4, m_cl_bandwidth[0]);
    k_logl_values_1d_mat.setArg(5, static_cast<CType>(m_lognorm_const));
    k_logl_values_1d_mat.setArg(6, output_mat);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_logl_values_1d_mat, cl::NullRange, cl::NDRange(N * test_length), cl::NullRange));

    auto& k_add_logl_values_1d_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::add_logl_values_1d_mat);
    k_add_logl_values_1d_mat.setArg(1, static_cast<unsigned int>(N));
    k_add_logl_values_1d_mat.setArg(2, test_buffer);
    k_add_logl_values_1d_mat.setArg(5, output_mat);

    for (size_t i = 1; i < m_variables.size(); ++i) {
        k_add_logl_values_1d_mat.setArg(0, m_training[i]);
        k_add_logl_values_1d_mat.setArg(3, static_cast<unsigned int>(i * test_length) + test_offset);
        k_add_logl_values_1d_mat.setArg(4, m_cl_bandwidth[i]);
        RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
            k_add_logl_values_1d_mat, cl::NullRange, cl::NDRange(N * test_length), cl::NullRange));
    }
}

template <typename ArrowType>
cl::Buffer ProductKDE::_logl_impl(cl::Buffer& test_buffer, int m) const {
    using CType = typename ArrowType::c_type;
    auto& opencl = OpenCLConfig::get();
    auto res = opencl.new_buffer<CType>(m);

    auto [mat_logls, allocated_m] = opencl.allocate_temp_mat<ArrowType>(N, m);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    for (auto i = 0; i < (iterations - 1); ++i) {
        product_logl_mat<ArrowType>(test_buffer, i * allocated_m, allocated_m, mat_logls);
        opencl.logsumexp_cols_offset<ArrowType>(mat_logls, N, allocated_m, res, i * allocated_m);
    }

    auto remaining_m = m - (iterations - 1) * allocated_m;
    product_logl_mat<ArrowType>(test_buffer, m - remaining_m, remaining_m, mat_logls);
    opencl.logsumexp_cols_offset<ArrowType>(mat_logls, N, remaining_m, res, m - remaining_m);

    return res;
}

template <typename ArrowType>
double ProductKDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto logl_buff = logl_buffer<ArrowType>(df);
    auto m = df.valid_rows(m_variables);

    auto& opencl = OpenCLConfig::get();
    auto buffer_sum = opencl.sum1d<ArrowType>(logl_buff, m);

    CType result = 0;
    opencl.read_from_buffer(&result, buffer_sum, 1);
    return static_cast<double>(result);
}

template <typename ArrowType>
py::tuple ProductKDE::__getstate__() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    VectorXd bw;
    std::vector<VectorType> training_data;
    double lognorm_const = -1;
    int N_export = -1;
    int training_type = -1;

    if (m_fitted) {
        auto& opencl = OpenCLConfig::get();

        for (size_t i = 0; i < m_variables.size(); ++i) {
            VectorType column(N);
            opencl.read_from_buffer(column.data(), m_training[i], N);
        }

        lognorm_const = m_lognorm_const;
        training_type = static_cast<int>(m_training_type->id());
        N_export = N;
        bw = m_bandwidth;
    }

    return py::make_tuple(
        m_variables, m_fitted, m_bselector, bw, training_data, lognorm_const, N_export, training_type);
}

}  // namespace kde

#endif  // PYBNESIAN_KDE_PRODUCTKDE_HPP