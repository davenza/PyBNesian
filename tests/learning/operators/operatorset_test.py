import pytest
import pyarrow as pa
from pgm_dataset.learning.operators import ArcOperatorSet, ChangeNodeTypeSet
from pgm_dataset.learning.scores import BIC, CVLikelihood
from pgm_dataset.models import GaussianNetwork, SemiparametricBN
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)


def test_create():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])

    bic = BIC(df)
    arc_gbn_bic = ArcOperatorSet(gbn, bic, [], [], 0)

    arc_gbn_bic.cache_scores(gbn)
    op = arc_gbn_bic.find_max(gbn)
    arc_gbn_bic.update_scores(gbn, op)

    # cv = CVLikelihood(df)
    # arc_gbn_cv = ArcOperatorSet(gbn, cv, [], [], 0)



    # with pytest.raises(TypeError) as ex:
    #     node_gbn = ChangeNodeTypeSet(gbn, bic, [], [], 0)
    # "incompatible function arguments." in str(ex.value)



    # pass