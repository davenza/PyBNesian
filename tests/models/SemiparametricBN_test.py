import pytest
import numpy as np
from pybnesian.factors import FactorType
from pybnesian.factors.continuous import LinearGaussianCPD, CKDE, SemiparametricCPD
from pybnesian.models import SemiparametricBN
import util_test

df = util_test.generate_normal_data(10000)

def test_create_spbn():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == FactorType.LinearGaussianCPD

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == FactorType.LinearGaussianCPD

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == FactorType.LinearGaussianCPD

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'c', 'b')])

    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'd')])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "must be a DAG" in str(ex.value)


    expected_node_type = {'a': FactorType.CKDE, 'b': FactorType.LinearGaussianCPD, 'c': FactorType.CKDE, 'd': FactorType.LinearGaussianCPD}

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'c', 'b')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])

    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'd')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([('a', 'b'), ('b', 'c'), ('c', 'a')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')], [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    assert "must be a DAG" in str(ex.value)


def test_node_type():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == FactorType.LinearGaussianCPD
    
    for i in range(spbn.num_nodes()):
        assert spbn.node_type(i) == FactorType.LinearGaussianCPD

    spbn.set_node_type('b', FactorType.CKDE)
    assert spbn.node_type('b') == FactorType.CKDE
    spbn.set_node_type('b', FactorType.LinearGaussianCPD)
    assert spbn.node_type('b') == FactorType.LinearGaussianCPD

    spbn.set_node_type(2, FactorType.CKDE)
    assert spbn.node_type(2) == FactorType.CKDE
    spbn.set_node_type(2, FactorType.LinearGaussianCPD)
    assert spbn.node_type(2) == FactorType.LinearGaussianCPD

def test_fit():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        for n in spbn.nodes():
            cpd = spbn.cpd(n)
    assert "not added" in str(ex.value)

    spbn.fit(df)

    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        assert cpd.factor_type == FactorType.LinearGaussianCPD

        lg = cpd.as_lg()
        assert type(lg) == LinearGaussianCPD
        assert lg.variable == n
        assert lg.evidence == spbn.parents(n)

    spbn.fit(df)
    
    spbn.remove_arc('a', 'b')

    cpd_b = spbn.cpd('b')
    lg_b = cpd_b.as_lg()
    assert lg_b.evidence != spbn.parents('b')

    spbn.fit(df)
    cpd_b = spbn.cpd('b')
    lg_b = cpd_b.as_lg()
    assert lg_b.evidence == spbn.parents('b')

    spbn.set_node_type('c', FactorType.CKDE)

    cpd_c = spbn.cpd('c')
    assert cpd_c.factor_type != spbn.node_type('c')

    spbn.fit(df)
    cpd_c = spbn.cpd('c')
    assert cpd_c.factor_type == spbn.node_type('c')


def test_cpd():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], [('d', FactorType.CKDE)])

    with pytest.raises(ValueError) as ex:
        spbn.cpd('a')
    assert "not added" in str(ex.value)

    spbn.fit(df)

    assert hex(id(spbn.cpd('a'))) == hex(id(spbn.cpd('a')))
    assert hex(id(spbn.cpd('b'))) == hex(id(spbn.cpd('b')))
    assert hex(id(spbn.cpd('c'))) == hex(id(spbn.cpd('c')))
    assert hex(id(spbn.cpd('d'))) == hex(id(spbn.cpd('d')))

    # Its conversion is also a reference.
    assert hex(id(spbn.cpd('a').as_lg())) == hex(id(spbn.cpd('a').as_lg()))
    assert hex(id(spbn.cpd('b').as_lg())) == hex(id(spbn.cpd('b').as_lg()))
    assert hex(id(spbn.cpd('c').as_lg())) == hex(id(spbn.cpd('c').as_lg()))
    assert hex(id(spbn.cpd('d').as_ckde())) == hex(id(spbn.cpd('d').as_ckde()))


def test_add_cpds():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], [('d', FactorType.CKDE)])

    with pytest.raises(ValueError) as ex:
        spbn.add_cpds([CKDE('a', [])])
    "CPD defined with a different node type" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn.add_cpds([LinearGaussianCPD('d', ['a', 'b', 'c'])])
    "CPD defined with a different node type" in str(ex.value)

    lg = LinearGaussianCPD('b', ['a'], [2.5, 1.65], 4)
    ckde = CKDE('d', ['a', 'b', 'c'])
    assert lg.fitted
    assert not ckde.fitted

    spbn.add_cpds([lg, ckde])

    assert not spbn.cpd('a').fitted
    assert spbn.cpd('b').fitted
    assert not spbn.cpd('c').fitted
    assert not spbn.cpd('d').fitted

def test_logl():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    spbn.fit(df)

    test_df = util_test.generate_normal_data(5000)
    ll = spbn.logl(test_df)
    sll = spbn.slogl(test_df)

    sum_ll = np.zeros((5000,))
    sum_sll = 0
    
    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        l = cpd.logl(test_df)
        s = cpd.slogl(test_df)
        assert np.all(np.isclose(s, l.sum()))
        sum_ll += l
        sum_sll += s
    
    assert np.all(np.isclose(ll, sum_ll))
    assert np.isclose(sll, ll.sum())
    assert sll == sum_sll