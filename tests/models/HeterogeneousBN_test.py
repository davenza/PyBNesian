import pybnesian as pbn
import pyarrow as pa


def test_type_equality():
    # 
    # Test single vector types
    # 

    het_single = pbn.HeterogeneousBN([pbn.CKDEType(), pbn.LinearGaussianCPDType()], ["a", "b", "c", "d"])
    het2_single = pbn.HeterogeneousBN([pbn.CKDEType(), pbn.LinearGaussianCPDType()], ["a", "b", "c", "d"])

    assert het_single.type() == het2_single.type()

    het3_single = pbn.HeterogeneousBN([pbn.LinearGaussianCPDType(), pbn.CKDEType()], ["a", "b", "c", "d"])
    
    assert het_single.type() != het3_single.type()

    # 
    # Test a single vector type for each data type
    # 

    het_dt = pbn.HeterogeneousBN({
        pa.float64(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()],
        pa.float32(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()],
        pa.dictionary(pa.int8(), pa.string()): [pbn.DiscreteFactorType()]
    }, ["a", "b", "c", "d"])

    het2_dt = pbn.HeterogeneousBN({
        pa.dictionary(pa.int8(), pa.string()): [pbn.DiscreteFactorType()],
        pa.float32(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()],
        pa.float64(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()]
    }, ["a", "b", "c", "d"])
    
    # The order of the set is not relevant
    assert het_dt.type() == het2_dt.type()

    het3_dt = pbn.HeterogeneousBN({
        pa.dictionary(pa.int8(), pa.string()): [pbn.DiscreteFactorType()],
        pa.float32(): [pbn.LinearGaussianCPDType(), pbn.CKDEType()],
        pa.float64(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()]
    }, ["a", "b", "c", "d"])

    # The order of the default FactorTypes is relevant
    assert het_dt.type() != het3_dt.type()
    
    # 
    # Compare single vector and multi vector FactorTypes

    het_single = pbn.HeterogeneousBN([pbn.CKDEType(), pbn.LinearGaussianCPDType()], ["a", "b", "c", "d"])
    het_dt = pbn.HeterogeneousBN({
        pa.float64(): [pbn.CKDEType(), pbn.LinearGaussianCPDType()]
    }, ["a", "b", "c", "d"])

    assert het_single.type() != het_dt.type()
