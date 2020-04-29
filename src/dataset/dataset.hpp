//
// Created by david on 17/4/20.
//

#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>

namespace py = pybind11;

using Eigen::MatrixXd;

namespace dataset {
    typedef py::handle PyDataset;

    bool is_pandas_dataframe(py::handle pyobject);

    std::shared_ptr<arrow::RecordBatch> to_record_batch(py::handle pyobject);

    class DataFrame {
    public:
        DataFrame(std::shared_ptr<arrow::RecordBatch> rb);

        int64_t null_count();
        int64_t null_instances_count();
        DataFrame drop_null_instances();

        MatrixXd to_eigen();

        std::shared_ptr<arrow::RecordBatch> operator->();
    private:
        std::shared_ptr<arrow::Buffer> combined_bitmap();
        std::shared_ptr<arrow::Buffer> combined_bitmap_with_null();


        std::shared_ptr<arrow::RecordBatch> m_batch;
    };






}




#endif //PGM_DATASET_DATASET_HPP
