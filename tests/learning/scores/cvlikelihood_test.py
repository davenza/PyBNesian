import pyarrow as pa
from pgm_dataset.models import GaussianNetwork, SemiparametricBN, NodeType
from pgm_dataset.learning.scores import CVLikelihood
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_score_gbn():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    cv = CVLikelihood(df, 10, 0)

    assert cv.score(gbn) == (
                            cv.local_score(gbn, 'a', []) +
                            cv.local_score(gbn, 'b', ['a']) +
                            cv.local_score(gbn, 'c', ['a', 'b']) +
                            cv.local_score(gbn, 'd', ['a', 'b', 'c']))

    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], 
                            [('a', NodeType.CKDE), ('c', NodeType.CKDE)])

    assert cv.score(spbn) == (
                            cv.local_score(spbn, 'a', []) +
                            cv.local_score(spbn, 'b', ['a']) +
                            cv.local_score(spbn, 'c', ['a', 'b']) +
                            cv.local_score(spbn, 'd', ['a', 'b', 'c']))
