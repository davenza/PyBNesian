import pyarrow as pa
from pgm_dataset.graph import PartiallyDirectedGraph, DirectedGraph, UndirectedGraph, Dag
import pickle


dig = DirectedGraph(['a', 'b', 'c'], [('a', 'c'), ('b', 'c')])

with open('digraph.pkl', 'wb') as f:
    pickle.dump(dig, f, 2)

with open('digraph.pkl', 'rb') as f:
    same_g = pickle.load(f)

print(same_g.nodes())
print(same_g.arcs())


ung = UndirectedGraph(['a', 'b', 'c'], [('a', 'c'), ('b', 'c')])

with open('ungraph.pkl', 'wb') as f:
    pickle.dump(ung, f, 2)

with open('ungraph.pkl', 'rb') as f:
    same_g = pickle.load(f)

print(same_g.nodes())
print(same_g.edges())


