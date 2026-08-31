import pybnesian as pbn


def test_meek_rule1():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr1 = pbn.PartiallyDirectedGraph(["X", "Y", "Z"], [("X", "Y")], [("Y", "Z")])

    assert pbn.MeekRules.rule1(gr1)
    assert gr1.num_edges() == 0
    assert set(gr1.arcs()) == set([("X", "Y"), ("Y", "Z")])

    assert not pbn.MeekRules.rule1(gr1)


def test_meek_rule2():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr2 = pbn.PartiallyDirectedGraph(
        ["X", "Y", "Z"], [("X", "Y"), ("Y", "Z")], [("X", "Z")]
    )

    assert pbn.MeekRules.rule2(gr2)
    assert gr2.num_edges() == 0
    assert set(gr2.arcs()) == set([("X", "Y"), ("Y", "Z"), ("X", "Z")])
    assert not pbn.MeekRules.rule2(gr2)


def test_meek_rule3():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr3 = pbn.PartiallyDirectedGraph(
        ["X", "Y1", "Y2", "Z"],
        [("Y1", "Z"), ("Y2", "Z")],
        [("X", "Y1"), ("X", "Y2"), ("X", "Z")],
    )

    assert pbn.MeekRules.rule3(gr3)
    assert set(gr3.edges()) == set([("X", "Y1"), ("X", "Y2")])
    assert set(gr3.arcs()) == set([("X", "Z"), ("Y1", "Z"), ("Y2", "Z")])
    assert not pbn.MeekRules.rule3(gr3)


def test_meek_sequential():
    # From Koller Chapter 3.4, Figure 3.13, pag 90.
    koller = pbn.PartiallyDirectedGraph(
        ["a", "b", "c", "d", "E", "F", "G"],
        [("b", "E"), ("c", "E")],
        [("a", "b"), ("b", "d"), ("c", "F"), ("E", "F"), ("F", "G")],
    )
    changed = True
    while changed:
        changed = False
        changed = changed or pbn.MeekRules.rule1(koller)
        changed = changed or pbn.MeekRules.rule2(koller)
        changed = changed or pbn.MeekRules.rule3(koller)

    assert set(koller.edges()) == set([("a", "b"), ("b", "d")])
    assert set(koller.arcs()) == set(
        [("b", "E"), ("c", "E"), ("E", "F"), ("c", "F"), ("F", "G")]
    )
