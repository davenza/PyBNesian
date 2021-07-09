import pybnesian as pbn

def test_OperatorTabuSet():
    tabu_set = pbn.OperatorTabuSet()

    assert tabu_set.empty()

    assert not tabu_set.contains(pbn.AddArc("a", "b", 1))    
    tabu_set.insert(pbn.AddArc("a", "b", 2))
    assert not tabu_set.empty()
    assert tabu_set.contains(pbn.AddArc("a", "b", 3))    

    assert not tabu_set.contains(pbn.RemoveArc("b", "c", 4))
    tabu_set.insert(pbn.RemoveArc("b", "c", 5))
    assert tabu_set.contains(pbn.RemoveArc("b", "c", 6))

    assert not tabu_set.contains(pbn.FlipArc("c", "d", 7))
    tabu_set.insert(pbn.RemoveArc("c", "d", 8))
    assert tabu_set.contains(pbn.RemoveArc("c", "d", 9))

    tabu_set.clear()
    assert tabu_set.empty()


