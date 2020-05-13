import pyarrow as pa
from pgm_dataset import AdjMatrixDag, AdjListDag




m = AdjMatrixDag(5)
m.add_node("a")
m.print()
print()
l = AdjListDag(5)
l.add_node("a")
l.print()