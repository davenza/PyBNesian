#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>

namespace py = pybind11;

using Eigen::MatrixXd;

typedef std::shared_ptr<arrow::Array> Array_ptr;

namespace dataset {
    typedef py::handle PyDataset;

    bool is_pandas_dataframe(py::handle pyobject);

    std::shared_ptr<arrow::RecordBatch> to_record_batch(py::handle pyobject);

    class DataFrame {
    public:
        DataFrame(std::shared_ptr <arrow::RecordBatch> rb);

        int64_t null_count() const;
        int64_t null_instances_count() const;
        DataFrame drop_null_instances();
        Array_ptr loc(int i) const;
        Array_ptr loc(const std::string &name) const;
        MatrixXd to_eigen() const;
        std::shared_ptr<arrow::RecordBatch> operator->();
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap() const;
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;
        std::shared_ptr <arrow::RecordBatch> m_batch;
    };
}


#endif //PGM_DATASET_DATASET_HPP
