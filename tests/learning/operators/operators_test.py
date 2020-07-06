import pyarrow as pa
from pgm_dataset.models import BayesianNetworkType
from pgm_dataset.learning.operators import AddArc


def test_create():
    o = AddArc(BayesianNetworkType.GBN, "a", "b", 0)

    print(o.source())
    pass