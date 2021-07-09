import pytest
import numpy as np
import pybnesian as pbn
from pybnesian import BayesianNetwork, GaussianNetwork
import util_test

df = util_test.generate_normal_data(10000)

def test_create_bn():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])

    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 0
    assert gbn.nodes() == ['a', 'b', 'c', 'd']

    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'c')])
    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 1
    assert gbn.nodes() == ['a', 'b', 'c', 'd']

    gbn = GaussianNetwork([('a', 'c'), ('b', 'd'), ('c', 'd')])
    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 3
    assert gbn.nodes() == ['a', 'c', 'b', 'd']

    with pytest.raises(TypeError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c'], [('a', 'c', 'b')])
    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c'], [('a', 'd')])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork([('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = BayesianNetwork(pbn.GaussianNetworkType(), ['a', 'b', 'c', 'd'], [], [('a', pbn.CKDEType())])
    assert "Wrong factor type" in str(ex.value)
    
def gbn_generator():
    # Test different Networks created with different constructors.
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    yield gbn
    gbn = GaussianNetwork([('a', 'c'), ('b', 'd'), ('c', 'd')])
    yield gbn
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c')])
    yield gbn

def test_nodes_util():
    for gbn in gbn_generator():
        assert gbn.num_nodes() == 4

        nodes = gbn.nodes()
        indices = gbn.indices()

        assert nodes[gbn.index('a')] == 'a'
        assert nodes[gbn.index('b')] == 'b'
        assert nodes[gbn.index('c')] == 'c'
        assert nodes[gbn.index('d')] == 'd'

        assert indices[gbn.name(0)] == 0
        assert indices[gbn.name(1)] == 1
        assert indices[gbn.name(2)] == 2
        assert indices[gbn.name(3)] == 3

        assert gbn.contains_node('a')
        assert gbn.contains_node('b')
        assert gbn.contains_node('c')
        assert gbn.contains_node('d')
        assert not gbn.contains_node('e')

def test_parent_children():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    
    assert gbn.num_parents('a') == 0
    assert gbn.num_parents('b') == 0
    assert gbn.num_parents('c') == 0
    assert gbn.num_parents('d') == 0

    assert gbn.parents('a') == []
    assert gbn.parents('b') == []
    assert gbn.parents('c') == []
    assert gbn.parents('d') == []

    assert gbn.num_children('a') == 0
    assert gbn.num_children('b') == 0
    assert gbn.num_children('c') == 0
    assert gbn.num_children('d') == 0

    gbn = GaussianNetwork([('a', 'c'), ('b', 'd'), ('c', 'd')])

    assert gbn.num_parents('a') == 0
    assert gbn.num_parents('b') == 0
    assert gbn.num_parents('c') == 1
    assert gbn.num_parents('d') == 2

    assert gbn.parents('a') == []
    assert gbn.parents('b') == []
    assert gbn.parents('c') == ['a']
    assert set(gbn.parents('d')) == set(['b', 'c'])

    assert gbn.num_children('a') == 1
    assert gbn.num_children('b') == 1
    assert gbn.num_children('c') == 1
    assert gbn.num_children('d') == 0

    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c')])

    assert gbn.num_parents('a') == 0
    assert gbn.num_parents('b') == 1
    assert gbn.num_parents('c') == 1
    assert gbn.num_parents('d') == 0

    assert gbn.parents('a') == []
    assert gbn.parents('b') == ['a']
    assert gbn.parents('c') == ['b']
    assert gbn.parents('d') == []

    assert gbn.num_children('a') == 1
    assert gbn.num_children('b') == 1
    assert gbn.num_children('c') == 0
    assert gbn.num_children('d') == 0

def test_arcs():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])

    assert gbn.num_arcs() == 0
    assert gbn.arcs() == []
    assert not gbn.has_arc('a', 'b')

    gbn.add_arc('a', 'b')
    assert gbn.num_arcs() == 1
    assert gbn.arcs() == [('a', 'b')]
    assert gbn.parents('b') == ['a']
    assert gbn.num_parents('b') == 1
    assert gbn.num_children('a') == 1
    assert gbn.has_arc('a', 'b')

    gbn.add_arc('b', 'c')
    assert gbn.num_arcs() == 2
    assert set(gbn.arcs()) == set([('a', 'b'), ('b', 'c')])
    assert gbn.parents('c') == ['b']
    assert gbn.num_parents('c') == 1
    assert gbn.num_children('b') == 1
    assert gbn.has_arc('b', 'c')
    
    gbn.add_arc('d', 'c')
    assert gbn.num_arcs() == 3
    assert set(gbn.arcs()) == set([('a', 'b'), ('b', 'c'), ('d', 'c')])
    assert set(gbn.parents('c')) == set(['b', 'd'])
    assert gbn.num_parents('c') == 2
    assert gbn.num_children('d') == 1
    assert gbn.has_arc('d', 'c')

    assert gbn.has_path('a', 'c')
    assert not gbn.has_path('a', 'd')
    assert gbn.has_path('b', 'c')
    assert gbn.has_path('d', 'c')

    assert not gbn.can_add_arc('c', 'a')
    # This edge exists, but virtually we consider that the addition is allowed. 
    assert gbn.can_add_arc('b', 'c')
    assert gbn.can_add_arc('d', 'a')

    gbn.add_arc('b', 'd')
    assert gbn.num_arcs() == 4
    assert set(gbn.arcs()) == set([('a', 'b'), ('b', 'c'), ('d', 'c'), ('b', 'd')])
    assert gbn.parents('d') == ['b']
    assert gbn.num_parents('d') == 1
    assert gbn.num_children('b') == 2
    assert gbn.has_arc('b', 'd')

    assert gbn.has_path('a', 'd')
    assert not gbn.can_add_arc('d', 'a')
    assert not gbn.can_flip_arc('b', 'c')
    assert gbn.can_flip_arc('a', 'b')
    # This edge does not exist, but it could be flipped if it did.
    assert gbn.can_flip_arc('d', 'a')

    # We can add an edge twice without changes.
    gbn.add_arc('b', 'd')
    assert gbn.num_arcs() == 4
    assert set(gbn.arcs()) == set([('a', 'b'), ('b', 'c'), ('d', 'c'), ('b', 'd')])
    assert gbn.parents('d') == ['b']
    assert gbn.num_parents('d') == 1
    assert gbn.num_children('b') == 2
    assert gbn.has_arc('b', 'd')

    gbn.remove_arc('b', 'c')
    assert gbn.num_arcs() == 3
    assert set(gbn.arcs()) == set([('a', 'b'), ('d', 'c'), ('b', 'd')])
    assert gbn.parents('c') == ['d']
    assert gbn.num_parents('c') == 1
    assert gbn.num_children('b') == 1
    assert not gbn.has_arc('b', 'c')

    assert gbn.can_add_arc('b', 'c')
    assert not gbn.can_add_arc('c', 'b')
    assert gbn.has_path('a', 'c')
    assert gbn.has_path('b', 'c')

    gbn.remove_arc('d', 'c')    
    assert gbn.num_arcs() == 2
    assert set(gbn.arcs()) == set([('a', 'b'), ('b', 'd')])
    assert gbn.parents('c') == []
    assert gbn.num_parents('c') == 0
    assert gbn.num_children('d') == 0
    assert not gbn.has_arc('d', 'c')

    assert gbn.can_add_arc('b', 'c')
    assert gbn.can_add_arc('c', 'b')
    assert not gbn.has_path('a', 'c')
    assert not gbn.has_path('b', 'c')

def test_bn_fit():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        for n in gbn.nodes():
            cpd = gbn.cpd(n)
    assert "not added" in str(ex.value)

    gbn.fit(df)

    for n in gbn.nodes():
        cpd = gbn.cpd(n)
        assert cpd.variable() == n
        assert cpd.evidence() == gbn.parents(n)

    gbn.fit(df)
    
    gbn.remove_arc('a', 'b')

    cpd_b = gbn.cpd('b')
    assert cpd_b.evidence != gbn.parents('b')

    gbn.fit(df)

    cpd_b = gbn.cpd('b')
    assert cpd_b.evidence() == gbn.parents('b')

def test_add_cpds():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('e', [])])
    assert "variable which is not present" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('a', ['e'])])
    assert "Evidence variable" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('a', ['b'])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('b', [])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('b', ['c'])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    lg = pbn.LinearGaussianCPD('b', ['a'], [2.5, 1.65], 4)
    assert lg.fitted()

    gbn.add_cpds([lg])

    cpd_b = gbn.cpd('b')
    assert cpd_b.variable() == 'b'
    assert cpd_b.evidence() == ['a']
    assert cpd_b.fitted()
    assert np.all(cpd_b.beta == np.asarray([2.5, 1.65]))
    assert cpd_b.variance == 4

    with pytest.raises(ValueError) as ex:
        cpd_a = gbn.cpd('a')
    assert "CPD of variable \"a\" not added. Call add_cpds() or fit() to add the CPD." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        cpd_c = gbn.cpd('c')
    assert "CPD of variable \"c\" not added. Call add_cpds() or fit() to add the CPD." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        cpd_d = gbn.cpd('d')
    assert "CPD of variable \"d\" not added. Call add_cpds() or fit() to add the CPD." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD('e', [])])
    assert "variable which is not present" in str(ex.value)


def test_bn_logl():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    gbn.fit(df)

    test_df = util_test.generate_normal_data(5000)
    ll = gbn.logl(test_df)
    sll = gbn.slogl(test_df)

    sum_ll = np.zeros((5000,))
    sum_sll = 0
    
    for n in gbn.nodes():
        cpd = gbn.cpd(n)
        l = cpd.logl(test_df)
        s = cpd.slogl(test_df)
        assert np.all(np.isclose(s, l.sum()))
        sum_ll += l
        sum_sll += s
    
    assert np.all(np.isclose(ll, sum_ll))
    assert np.isclose(sll, ll.sum())
    assert sll == sum_sll

def test_bn_sample():
    gbn = GaussianNetwork(['a', 'c', 'b', 'd'], [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    gbn.fit(df)
    sample = gbn.sample(1000, 0, False)

    # Not ordered, so topological sort.
    assert sample.schema.names == ['a', 'b', 'c', 'd']
    assert sample.num_rows == 1000
    
    sample_ordered = gbn.sample(1000, 0, True)
    assert sample_ordered.schema.names == ['a', 'c', 'b', 'd']
    assert sample_ordered.num_rows == 1000

    assert sample.column(0).equals(sample_ordered.column(0))
    assert sample.column(1).equals(sample_ordered.column(2))
    assert sample.column(2).equals(sample_ordered.column(1))
    assert sample.column(3).equals(sample_ordered.column(3))

    other_seed = gbn.sample(1000, 1, False)

    assert not sample.column(0).equals(other_seed.column(0))
    assert not sample.column(1).equals(other_seed.column(2))
    assert not sample.column(2).equals(other_seed.column(1))
    assert not sample.column(3).equals(other_seed.column(3))
