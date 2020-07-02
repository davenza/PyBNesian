import pytest
import pyarrow as pa
import numpy as np
from pgm_dataset.models import GaussianNetwork
from pgm_dataset.factors.continuous import LinearGaussianCPD
import util_test

df = util_test.generate_normal_data(10000)

def test_create_bn():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])

    assert gbn.num_nodes() == 4
    assert gbn.num_edges() == 0
    assert gbn.nodes() == ['a', 'b', 'c', 'd']

    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'c')])
    assert gbn.num_nodes() == 4
    assert gbn.num_edges() == 1
    assert gbn.nodes() == ['a', 'b', 'c', 'd']

    gbn = GaussianNetwork([('a', 'c'), ('b', 'd'), ('c', 'd')])
    assert gbn.num_nodes() == 4
    assert gbn.num_edges() == 3
    assert gbn.nodes() == ['a', 'c', 'b', 'd']

    with pytest.raises(TypeError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c'], [('a', 'c', 'b')])

    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c'], [('a', 'd')])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork([('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "The graph must be a DAG." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "The graph must be a DAG." in str(ex.value)
    
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

    assert gbn.parent_indices('a') == []
    assert gbn.parent_indices('b') == []
    assert gbn.parent_indices('c') == []
    assert gbn.parent_indices('d') == []

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

    assert gbn.parent_indices('a') == []
    assert gbn.parent_indices('b') == []
    assert gbn.parent_indices('c') == [0]
    assert set(gbn.parent_indices('d')) == set([2, 1])

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

    assert gbn.parent_indices('a') == []
    assert gbn.parent_indices('b') == [0]
    assert gbn.parent_indices('c') == [1]
    assert gbn.parent_indices('d') == []

    assert gbn.num_children('a') == 1
    assert gbn.num_children('b') == 1
    assert gbn.num_children('c') == 0
    assert gbn.num_children('d') == 0

def test_edges():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])

    assert gbn.num_edges() == 0
    assert not gbn.has_edge('a', 'b')

    gbn.add_edge('a', 'b')
    assert gbn.num_edges() == 1
    assert gbn.parents('b') == ['a']
    assert gbn.num_parents('b') == 1
    assert gbn.num_children('a') == 1
    assert gbn.has_edge('a', 'b')

    gbn.add_edge('b', 'c')
    assert gbn.num_edges() == 2
    assert gbn.parents('c') == ['b']
    assert gbn.num_parents('c') == 1
    assert gbn.num_children('b') == 1
    assert gbn.has_edge('b', 'c')
    
    gbn.add_edge('d', 'c')
    assert gbn.num_edges() == 3
    assert set(gbn.parents('c')) == set(['b', 'd'])
    assert gbn.num_parents('c') == 2
    assert gbn.num_children('d') == 1
    assert gbn.has_edge('d', 'c')

    assert gbn.has_path('a', 'c')
    assert not gbn.has_path('a', 'd')
    assert gbn.has_path('b', 'c')
    assert gbn.has_path('d', 'c')

    assert not gbn.can_add_edge('c', 'a')
    # This edge exists, but virtually we consider that the addition is allowed. 
    assert gbn.can_add_edge('b', 'c')
    assert gbn.can_add_edge('d', 'a')

    gbn.add_edge('b', 'd')
    assert gbn.num_edges() == 4
    assert gbn.parents('d') == ['b']
    assert gbn.num_parents('d') == 1
    assert gbn.num_children('b') == 2
    assert gbn.has_edge('b', 'd')

    assert gbn.has_path('a', 'd')
    assert not gbn.can_add_edge('d', 'a')
    assert not gbn.can_flip_edge('b', 'c')
    assert gbn.can_flip_edge('a', 'b')
    # This edge does not exist, but it could be flipped if it did.
    assert gbn.can_flip_edge('d', 'a')

    # We can add an edge twice without chages.
    gbn.add_edge('b', 'd')
    assert gbn.num_edges() == 4
    assert gbn.parents('d') == ['b']
    assert gbn.num_parents('d') == 1
    assert gbn.num_children('b') == 2
    assert gbn.has_edge('b', 'd')

    gbn.remove_edge('b', 'c')
    assert gbn.num_edges() == 3
    assert gbn.parents('c') == ['d']
    assert gbn.num_parents('c') == 1
    assert gbn.num_children('b') == 1
    assert not gbn.has_edge('b', 'c')

    assert gbn.can_add_edge('b', 'c')
    assert not gbn.can_add_edge('c', 'b')
    assert gbn.has_path('a', 'c')
    assert gbn.has_path('b', 'c')

    gbn.remove_edge('d', 'c')    
    assert gbn.num_edges() == 2
    assert gbn.parents('c') == []
    assert gbn.num_parents('c') == 0
    assert gbn.num_children('d') == 0
    assert not gbn.has_edge('d', 'c')

    assert gbn.can_add_edge('b', 'c')
    assert gbn.can_add_edge('c', 'b')
    assert not gbn.has_path('a', 'c')
    assert not gbn.has_path('b', 'c')

def test_fit():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        for n in gbn.nodes():
            cpd = gbn.cpd(n)
    assert "not added" in str(ex.value)

    gbn.fit(df)

    for n in gbn.nodes():
        cpd = gbn.cpd(n)
        assert cpd.variable == n
        assert cpd.evidence == gbn.parents(n)

    gbn.fit(df)
    
    gbn.remove_edge('a', 'b')

    cpd_b = gbn.cpd('b')
    assert cpd_b.evidence != gbn.parents('b')

    gbn.fit(df)

    cpd_b = gbn.cpd('b')
    assert cpd_b.evidence == gbn.parents('b')

def test_cpd():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        gbn.cpd('a')
    assert "not added" in str(ex.value)

    gbn.add_cpds([LinearGaussianCPD('b', ['a'], [2.5, 1.65], 4)])

    # Always return a reference
    cpd_a = gbn.cpd('a')
    # TODO: Is hex(id()) the correct way to check the validity of the reference ? Operator is ?
    address_a = hex(id(cpd_a))
    assert not cpd_a.fitted
    gbn.cpd('a').fit(df)
    assert cpd_a.fitted
    assert address_a == hex(id(cpd_a))

    cpd_b = gbn.cpd('b')
    address_b = hex(id(cpd_b))
    assert cpd_b.fitted
    gbn.cpd('b').fit(df)
    assert cpd_b.fitted
    assert address_b == hex(id(cpd_b))

    cpd_c = gbn.cpd('c')
    address_c = hex(id(cpd_c))
    assert not cpd_c.fitted
    gbn.cpd('c').fit(df)
    assert cpd_c.fitted
    assert address_c == hex(id(cpd_c))

    cpd_d = gbn.cpd('d')
    address_d = hex(id(cpd_d))
    assert not cpd_d.fitted
    gbn.cpd('d').fit(df)
    assert cpd_d.fitted
    assert address_d == hex(id(cpd_d))

def test_add_cpds():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('e', [])])
    assert "variable which is not present" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('a', ['e'])])
    assert "Evidence variable" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('a', ['b'])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('b', [])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('b', ['c'])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    lg = LinearGaussianCPD('b', ['a'], [2.5, 1.65], 4)
    assert lg.fitted

    gbn.add_cpds([lg])

    cpd_b = gbn.cpd('b')
    assert cpd_b.variable == 'b'
    assert cpd_b.evidence == ['a']
    assert cpd_b.fitted
    assert np.all(cpd_b.beta == np.asarray([2.5, 1.65]))
    assert cpd_b.variance == 4

    cpd_a = gbn.cpd('a')
    assert not cpd_a.fitted
    cpd_c = gbn.cpd('c')
    assert not cpd_c.fitted
    cpd_d = gbn.cpd('d')
    assert not cpd_d.fitted

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([LinearGaussianCPD('e', [])])
    assert "variable which is not present" in str(ex.value)