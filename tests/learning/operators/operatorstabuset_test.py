import pytest
import pyarrow as pa
from pgm_dataset.models import BayesianNetworkType
from pgm_dataset.learning.operators import OperatorTabuSet, AddArc, RemoveArc, FlipArc, RemoveArc


def test_OperatorTabuSet():
    tabu_set = OperatorTabuSet(BayesianNetworkType.GBN)

    assert tabu_set.empty()

    assert not tabu_set.contains(AddArc(BayesianNetworkType.GBN, "a", "b", 1))    
    tabu_set.insert(AddArc(BayesianNetworkType.GBN, "a", "b", 2))
    assert not tabu_set.empty()
    assert tabu_set.contains(AddArc(BayesianNetworkType.GBN, "a", "b", 3))    

    assert not tabu_set.contains(RemoveArc(BayesianNetworkType.GBN, "b", "c", 4))
    tabu_set.insert(RemoveArc(BayesianNetworkType.GBN, "b", "c", 5))
    assert tabu_set.contains(RemoveArc(BayesianNetworkType.GBN, "b", "c", 6))

    assert not tabu_set.contains(FlipArc(BayesianNetworkType.GBN, "c", "d", 7))
    tabu_set.insert(RemoveArc(BayesianNetworkType.GBN, "c", "d", 8))
    assert tabu_set.contains(RemoveArc(BayesianNetworkType.GBN, "c", "d", 9))

    with pytest.raises(TypeError) as ex:
        tabu_set.insert(AddArc(BayesianNetworkType.SPBN, "a", "b", 1))
    "incompatible function arguments" in str(ex.value)

    tabu_set = OperatorTabuSet(BayesianNetworkType.SPBN)

    assert tabu_set.empty()

    assert not tabu_set.contains(AddArc(BayesianNetworkType.SPBN, "a", "b", 1))    
    tabu_set.insert(AddArc(BayesianNetworkType.SPBN, "a", "b", 2))
    assert not tabu_set.empty()
    assert tabu_set.contains(AddArc(BayesianNetworkType.SPBN, "a", "b", 3))    

    assert not tabu_set.contains(RemoveArc(BayesianNetworkType.SPBN, "b", "c", 4))
    tabu_set.insert(RemoveArc(BayesianNetworkType.SPBN, "b", "c", 5))
    assert tabu_set.contains(RemoveArc(BayesianNetworkType.SPBN, "b", "c", 6))

    assert not tabu_set.contains(FlipArc(BayesianNetworkType.SPBN, "c", "d", 7))
    tabu_set.insert(RemoveArc(BayesianNetworkType.SPBN, "c", "d", 8))
    assert tabu_set.contains(RemoveArc(BayesianNetworkType.SPBN, "c", "d", 9))

    with pytest.raises(TypeError) as ex:
        tabu_set.insert(AddArc(BayesianNetworkType.GBN, "a", "b", 2))
    "incompatible function arguments" in str(ex.value)
