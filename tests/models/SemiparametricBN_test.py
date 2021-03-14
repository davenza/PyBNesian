import pytest
import numpy as np
from pybnesian.factors.continuous import LinearGaussianCPD, LinearGaussianCPDType, CKDE, CKDEType
from pybnesian.models import SemiparametricBN
import util_test

df = util_test.generate_normal_data(10000)

def test_create_spbn():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == LinearGaussianCPDType()

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == LinearGaussianCPDType()

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == LinearGaussianCPDType()

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


    expected_node_type = {'a': CKDEType(),
                          'b': LinearGaussianCPDType(),
                          'c': CKDEType(),
                          'd': LinearGaussianCPDType()}

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', CKDEType()), ('c', CKDEType())])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')], [('a', CKDEType()), ('c', CKDEType())])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')], [('a', CKDEType()), ('c', CKDEType())])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'c', 'b')], [('a', CKDEType()), ('c', CKDEType())])
    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'd')], [('a', CKDEType()), ('c', CKDEType())])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([('a', 'b'), ('b', 'c'), ('c', 'a')], [('a', CKDEType()), ('c', CKDEType())])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c', 'd'],
                                [('a', 'b'), ('b', 'c'), ('c', 'a')],
                                [('a', CKDEType()), ('c', CKDEType())])
    assert "must be a DAG" in str(ex.value)


def test_node_type():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == LinearGaussianCPDType()
    
    spbn.set_node_type('b', CKDEType())
    assert spbn.node_type('b') == CKDEType()
    spbn.set_node_type('b', LinearGaussianCPDType())
    assert spbn.node_type('b') == LinearGaussianCPDType()

def test_fit():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        for n in spbn.nodes():
            cpd = spbn.cpd(n)
    assert "not added" in str(ex.value)

    spbn.fit(df)

    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        assert cpd.type() == LinearGaussianCPDType()

        assert type(cpd) == LinearGaussianCPD
        assert cpd.variable == n
        assert set(cpd.evidence) == set(spbn.parents(n))

    spbn.fit(df)
    
    spbn.remove_arc('a', 'b')

    cpd_b = spbn.cpd('b')
    assert type(cpd_b) == LinearGaussianCPD
    assert cpd_b.evidence != spbn.parents('b')

    spbn.fit(df)
    cpd_b = spbn.cpd('b')
    assert type(cpd_b) == LinearGaussianCPD
    assert cpd_b.evidence == spbn.parents('b')

    spbn.set_node_type('c', CKDEType())

    cpd_c = spbn.cpd('c')
    assert cpd_c.type() != spbn.node_type('c')

    spbn.fit(df)
    cpd_c = spbn.cpd('c')
    assert cpd_c.type() == spbn.node_type('c')


def test_cpd():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], [('d', CKDEType())])

    with pytest.raises(ValueError) as ex:
        spbn.cpd('a')
    assert "not added" in str(ex.value)

    spbn.fit(df)

    assert spbn.cpd('a').type() == LinearGaussianCPDType()
    assert spbn.cpd('b').type() == LinearGaussianCPDType()
    assert spbn.cpd('c').type() == LinearGaussianCPDType()
    assert spbn.cpd('d').type() == CKDEType()

    assert spbn.cpd('a').fitted
    assert spbn.cpd('b').fitted
    assert spbn.cpd('c').fitted
    assert spbn.cpd('d').fitted

def test_add_cpds():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], [('d', CKDEType())])

    with pytest.raises(ValueError) as ex:
        spbn.add_cpds([CKDE('a', [])])
    assert "Bayesian network expects type" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn.add_cpds([LinearGaussianCPD('d', ['a', 'b', 'c'])])
    assert "Bayesian network expects type" in str(ex.value)

    lg = LinearGaussianCPD('b', ['a'], [2.5, 1.65], 4)
    ckde = CKDE('d', ['a', 'b', 'c'])
    assert lg.fitted
    assert not ckde.fitted

    spbn.add_cpds([lg, ckde])

    with pytest.raises(ValueError) as ex:
        not spbn.cpd('a').fitted
    assert "CPD of variable \"a\" not added. Call add_cpds() or fit() to add the CPD." in str(ex.value)

    assert spbn.cpd('b').fitted

    with pytest.raises(ValueError) as ex:
        not spbn.cpd('c').fitted
    assert "CPD of variable \"c\" not added. Call add_cpds() or fit() to add the CPD." in str(ex.value)

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