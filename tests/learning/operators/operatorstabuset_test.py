import pytest
import pyarrow as pa
from pybnesian.models import BayesianNetworkType
from pybnesian.learning.operators import OperatorTabuSet, AddArc, RemoveArc, FlipArc, RemoveArc


def test_OperatorTabuSet():
    tabu_set = OperatorTabuSet()

    assert tabu_set.empty()

    assert not tabu_set.contains(AddArc("a", "b", 1))    
    tabu_set.insert(AddArc("a", "b", 2))
    assert not tabu_set.empty()
    assert tabu_set.contains(AddArc("a", "b", 3))    

    assert not tabu_set.contains(RemoveArc("b", "c", 4))
    tabu_set.insert(RemoveArc("b", "c", 5))
    assert tabu_set.contains(RemoveArc("b", "c", 6))

    assert not tabu_set.contains(FlipArc("c", "d", 7))
    tabu_set.insert(RemoveArc("c", "d", 8))
    assert tabu_set.contains(RemoveArc("c", "d", 9))

    tabu_set.clear()
    assert tabu_set.empty()


