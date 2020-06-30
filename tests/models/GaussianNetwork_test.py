import pyarrow as pa
from pgm_dataset.models import GaussianNetwork, SemiparametricBN



def test_create_gbn():
    gbn = GaussianNetwork(['a', 'b'])

    assert gbn.num_nodes() == 2
    assert gbn.num_edges() == 0
    assert gbn.nodes() == ['a', 'b']


    gbn = GaussianNetwork(['a', 'b', 'c'], [('a', 'c')])
    assert gbn.num_nodes() == 3
    assert gbn.num_edges() == 1
    assert gbn.nodes() == ['a', 'b', 'c']
