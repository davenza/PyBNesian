import pyarrow as pa
from pgm_dataset.graph import PartiallyDirectedGraph, Dag

g = PartiallyDirectedGraph([('a', 'b'), ('b', 'c'), ('a', 'c'), ('d', 'c')], [('d', 'a')])
print("Nodes: " + str(g.nodes()))
print("Edges: " + str(g.edges()))
print("Arcs: " + str(g.arcs()))
dag = g.to_dag()
print("Nodes: " + str(dag.nodes()))
print("Arcs: " + str(dag.arcs()))

# g = PartiallyDirectedGraph(['a', 'b', 'c', 'd', 'e', 'f'], 
#                             [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')], 
#                             [])
# # g = PartiallyDirectedGraph(['a', 'b', 'c', 'd'], 
# #                             [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')], 
# #                             [])
# print("Nodes: " + str(g.nodes()))
# print("Edges: " + str(g.edges()))
# print("Arcs: " + str(g.arcs()))
# dag = g.to_dag()
# print("Nodes: " + str(dag.nodes()))
# print("Arcs: " + str(dag.arcs()))



# dag = Dag([('a', 'c'), ('b', 'c'), ('c', 'd'), ('b', 'e'), ('e', 'f')])
# print("DAG:")
# print("Nodes: " + str(dag.nodes()))
# print("Arcs: " + str(dag.arcs()))


pdag = dag.to_pdag()
print("PDAG:")
print("Nodes: " + str(pdag.nodes()))
print("Edges: " + str(pdag.edges()))
print("Arcs: " + str(pdag.arcs()))