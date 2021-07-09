import pytest
import re
import numpy as np
import pandas as pd
from scipy.stats import norm
import pybnesian as pbn
from pybnesian import GaussianNetwork, ConditionalGaussianNetwork, DynamicGaussianNetwork
import util_test

df = util_test.generate_normal_data(1000)

def test_create_dbn():
    variables = ["a", "b", "c", "d"]
    gbn = DynamicGaussianNetwork(variables, 2)

    assert gbn.markovian_order() == 2
    assert gbn.variables() == ["a", "b", "c", "d"]
    assert gbn.num_variables() == 4
    assert gbn.type() == pbn.GaussianNetworkType()

    transition_nodes = [v + "_t_0" for v in variables]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]

    assert set(gbn.static_bn().nodes()) == set(static_nodes)
    assert set(gbn.transition_bn().interface_nodes()) == set(static_nodes)
    assert set(gbn.transition_bn().nodes()) == set(transition_nodes)

    static_bn = GaussianNetwork(static_nodes)
    transition_bn = ConditionalGaussianNetwork(transition_nodes, static_nodes)

    gbn2 = DynamicGaussianNetwork(variables, 2, static_bn, transition_bn)

    wrong_transition_bn = pbn.ConditionalDiscreteBN(transition_nodes, static_nodes)

    with pytest.raises(ValueError) as ex:
        gbn3 = DynamicGaussianNetwork(variables, 2, static_bn, wrong_transition_bn)
    assert "Static and transition Bayesian networks do not have the same type" in str(ex.value)

    wrong_static_bn = pbn.DiscreteBN(static_nodes)
    with pytest.raises(ValueError) as ex:
        gbn4 = DynamicGaussianNetwork(variables, 2, wrong_static_bn, wrong_transition_bn)
    assert "Bayesian networks are not Gaussian." in str(ex.value)

def test_variable_operations_dbn():
    variables = ["a", "b", "c", "d"]
    gbn = DynamicGaussianNetwork(variables, 2)

    assert gbn.markovian_order() == 2
    assert gbn.variables() == ["a", "b", "c", "d"]
    assert gbn.num_variables() == 4

    assert gbn.contains_variable("a")
    assert gbn.contains_variable("b")
    assert gbn.contains_variable("c")
    assert gbn.contains_variable("d")

    gbn.add_variable("e")
    assert set(gbn.variables()) == set(["a", "b", "c", "d", "e"])
    assert gbn.num_variables() == 5

    assert set(gbn.static_bn().nodes()) == set([v + "_t_" + str(m) for v in variables + ["e"] for m in range(1, 3)])
    assert set(gbn.transition_bn().nodes()) == set([v + "_t_0" for v in variables + ["e"]])

    gbn.remove_variable("b")
    assert set(gbn.variables()) == set(["a", "c", "d", "e"])
    assert gbn.num_variables() == 4
    assert set(gbn.static_bn().nodes()) == set([v + "_t_" + str(m) for v in ["a", "c", "d", "e"] for m in range(1, 3)])
    assert set(gbn.transition_bn().nodes()) == set([v + "_t_0" for v in ["a", "c", "d", "e"]])


def test_fit_dbn():
    variables = ["a", "b", "c", "d"]
    gbn = DynamicGaussianNetwork(variables, 2)
    assert not gbn.fitted()
    assert not gbn.static_bn().fitted()
    assert not gbn.transition_bn().fitted()
    gbn.fit(df)
    assert gbn.fitted()

    ddf = pbn.DynamicDataFrame(df, 2)
    gbn2 = DynamicGaussianNetwork(variables, 2)
    gbn2.static_bn().fit(ddf.static_df())
    assert not gbn2.fitted()
    assert gbn2.static_bn().fitted()
    assert not gbn2.transition_bn().fitted()

    gbn2.transition_bn().fit(ddf.transition_df())
    assert gbn2.fitted()
    assert gbn2.static_bn().fitted()
    assert gbn2.transition_bn().fitted()

def lg_logl_row(row, variable, evidence, beta, variance):
    m = beta[0] + beta[1:].dot(row[evidence])
    return norm(m, np.sqrt(variance)).logpdf(row[variable])

def static_logl(dbn, test_data, index, variable):
    sl = test_data.head(dbn.markovian_order())

    node_name = variable + "_t_" + str(dbn.markovian_order() - index)
    cpd = dbn.static_bn().cpd(node_name)
    evidence = cpd.evidence()

    row_values = [sl.loc[index, variable]]
    for e in evidence:
        m = re.search('(.*)_t_(\\d+)', e)
        e_var = m[1]
        t = int(m[2])

        row_values.append(sl.loc[dbn.markovian_order()-t, e_var])

    r = pd.Series(data=row_values, index=[node_name] + evidence)

    return lg_logl_row(r, node_name, evidence, cpd.beta, cpd.variance)

def transition_logl(dbn, test_data, index, variable):
    node_name = variable + "_t_0"
    cpd = dbn.transition_bn().cpd(node_name)
    evidence = cpd.evidence()

    row_values = [test_data.loc[index, variable]]
    for e in evidence:
        m = re.search('(.*)_t_(\\d+)', e)
        e_var = m[1]
        t = int(m[2])

        row_values.append(test_data.loc[index-t, e_var])

    r = pd.Series(data=row_values, index=[node_name] + evidence)
    return lg_logl_row(r, node_name, evidence, cpd.beta, cpd.variance)


def numpy_logl(dbn, test_data):
    ll = np.zeros((test_data.shape[0],))

    for i in range(dbn.markovian_order()):
        for v in dbn.variables():
            ll[i] += static_logl(dbn, test_data, i, v)

    for i in range(dbn.markovian_order(), test_data.shape[0]):
        for v in dbn.variables():
            ll[i] += transition_logl(dbn, test_data, i, v)

    return ll

def test_logl_dbn():
    variables = ["a", "b", "c", "d"]

    static_bn = GaussianNetwork(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])
    static_bn = GaussianNetwork(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])
    gbn = DynamicGaussianNetwork(variables, 2)

    static_bn = gbn.static_bn()
    static_bn.add_arc("a_t_2", "c_t_2")
    static_bn.add_arc("b_t_2", "c_t_2")
    static_bn.add_arc("c_t_2", "d_t_2")
    static_bn.add_arc("a_t_1", "c_t_1")
    static_bn.add_arc("b_t_1", "c_t_1")
    static_bn.add_arc("c_t_1", "d_t_1")

    transition_bn = gbn.transition_bn()
    transition_bn.add_arc("a_t_2", "a_t_0")
    transition_bn.add_arc("b_t_2", "b_t_0")
    transition_bn.add_arc("c_t_2", "c_t_0")
    transition_bn.add_arc("d_t_2", "d_t_0")
    transition_bn.add_arc("a_t_1", "a_t_0")
    transition_bn.add_arc("b_t_1", "b_t_0")
    transition_bn.add_arc("c_t_1", "c_t_0")
    transition_bn.add_arc("d_t_1", "d_t_0")

    gbn.fit(df)

    test_df = util_test.generate_normal_data(100)
    ground_truth_ll = numpy_logl(gbn, util_test.generate_normal_data(100))
    ll = gbn.logl(test_df)
    assert np.all(np.isclose(ground_truth_ll, ll))

def test_slogl_dbn():
    variables = ["a", "b", "c", "d"]

    static_bn = GaussianNetwork(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])
    static_bn = GaussianNetwork(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])
    gbn = DynamicGaussianNetwork(variables, 2)

    static_bn = gbn.static_bn()
    static_bn.add_arc("a_t_2", "c_t_2")
    static_bn.add_arc("b_t_2", "c_t_2")
    static_bn.add_arc("c_t_2", "d_t_2")
    static_bn.add_arc("a_t_1", "c_t_1")
    static_bn.add_arc("b_t_1", "c_t_1")
    static_bn.add_arc("c_t_1", "d_t_1")

    transition_bn = gbn.transition_bn()
    transition_bn.add_arc("a_t_2", "a_t_0")
    transition_bn.add_arc("b_t_2", "b_t_0")
    transition_bn.add_arc("c_t_2", "c_t_0")
    transition_bn.add_arc("d_t_2", "d_t_0")
    transition_bn.add_arc("a_t_1", "a_t_0")
    transition_bn.add_arc("b_t_1", "b_t_0")
    transition_bn.add_arc("c_t_1", "c_t_0")
    transition_bn.add_arc("d_t_1", "d_t_0")

    gbn.fit(df)
    test_df = util_test.generate_normal_data(100)
    ll = numpy_logl(gbn, test_df)
    assert np.isclose(gbn.slogl(test_df), ll.sum())