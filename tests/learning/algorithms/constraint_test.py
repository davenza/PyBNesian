from pybnesian import PartiallyDirectedGraph, MeekRules

def test_meek_rule1():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr1 = PartiallyDirectedGraph(["X", "Y", "Z"], [("X", "Y")], [("Y", "Z")])

    assert MeekRules.rule1(gr1)
    assert gr1.num_edges() == 0
    assert set(gr1.arcs()) == set([('X', 'Y'), ('Y', 'Z')])

    assert not MeekRules.rule1(gr1)

def test_meek_rule2():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr2 = PartiallyDirectedGraph(["X", "Y", "Z"], [("X", "Y"), ("Y", "Z")], [("X", "Z")])

    assert MeekRules.rule2(gr2)
    assert gr2.num_edges() == 0
    assert set(gr2.arcs()) == set([('X', 'Y'), ('Y', 'Z'), ('X', 'Z')])
    assert not MeekRules.rule2(gr2)

def test_meek_rule3():
    # From Koller Chapter 3.4, Figure 3.12, pag 89.
    gr3 = PartiallyDirectedGraph(["X", "Y1", "Y2", "Z"], [("Y1", "Z"), ("Y2", "Z")], [("X", "Y1"), ("X", "Y2"), ("X", "Z")])

    assert MeekRules.rule3(gr3)
    assert set(gr3.edges()) == set([('X', 'Y1'), ('X', 'Y2')])
    assert set(gr3.arcs()) == set([('X', 'Z'), ('Y1', 'Z'), ('Y2', 'Z')])
    assert not MeekRules.rule3(gr3)

def test_meek_sequential():
    # From Koller Chapter 3.4, Figure 3.13, pag 90.
    koller = PartiallyDirectedGraph(["A", "B", "C", "D", "E", "F", "G"], 
                                    [("B", "E"), ("C", "E")],
                                    [("A", "B"), ("B", "D"), ("C", "F"), ("E", "F"), ("F", "G")])
    changed = True
    while changed:
        changed = False
        changed = changed or MeekRules.rule1(koller)
        changed = changed or MeekRules.rule2(koller)
        changed = changed or MeekRules.rule3(koller)

    assert set(koller.edges()) == set([('A', 'B'), ('B', 'D')])
    assert set(koller.arcs()) == set([('B', 'E'), ('C', 'E'), ('E', 'F'), ('C', 'F'), ('F', 'G')])