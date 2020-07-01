import pytest
import pyarrow as pa
from pgm_dataset.factors.continuous import LinearGaussianCPD, CKDE
from pgm_dataset.models import SemiparametricBN, NodeType
import util_test

df = util_test.generate_normal_data(10000)

def test_create_spbn():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == NodeType.LinearGaussianCPD

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == NodeType.LinearGaussianCPD

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == NodeType.LinearGaussianCPD

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'c', 'b')])

    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'd')])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "The graph must be a DAG." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    assert "The graph must be a DAG." in str(ex.value)


    expected_node_type = {'a': NodeType.CKDE, 'b': NodeType.LinearGaussianCPD, 'c': NodeType.CKDE, 'd': NodeType.LinearGaussianCPD}

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'c')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 1
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN([('a', 'c'), ('b', 'd'), ('c', 'd')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 3
    assert spbn.nodes() == ['a', 'c', 'b', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'c', 'b')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])

    assert "incompatible constructor arguments" in str(ex.value)
    
    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c'], [('a', 'd')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([('a', 'b'), ('b', 'c'), ('c', 'a')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert "The graph must be a DAG." in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(['a', 'b', 'c', 'd'], [('a', 'b'), ('b', 'c'), ('c', 'a')], [('a', NodeType.CKDE), ('c', NodeType.CKDE)])
    assert "The graph must be a DAG." in str(ex.value)


def test_node_type():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_nodes() == 4
    assert spbn.num_edges() == 0
    assert spbn.nodes() == ['a', 'b', 'c', 'd']

    for n in spbn.nodes():
        assert spbn.node_type(n) == NodeType.LinearGaussianCPD
    
    for i in range(spbn.num_nodes()):
        assert spbn.node_type(i) == NodeType.LinearGaussianCPD

    spbn.set_node_type('b', NodeType.CKDE)
    assert spbn.node_type('b') == NodeType.CKDE
    spbn.set_node_type('b', NodeType.LinearGaussianCPD)
    assert spbn.node_type('b') == NodeType.LinearGaussianCPD

    spbn.set_node_type(2, NodeType.CKDE)
    assert spbn.node_type(2) == NodeType.CKDE
    spbn.set_node_type(2, NodeType.LinearGaussianCPD)
    assert spbn.node_type(2) == NodeType.LinearGaussianCPD

def test_fit():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    with pytest.raises(ValueError) as ex:
        for n in spbn.nodes():
            cpd = spbn.cpd(n)
    assert "not fitted" in str(ex.value)

    spbn.fit(df)

    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        assert cpd.node_type() == NodeType.LinearGaussianCPD

        lg = cpd.as_lg()
        assert type(lg) == LinearGaussianCPD

    spbn.fit(df)