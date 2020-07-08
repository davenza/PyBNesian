# import pyarrow as pa
# from pgm_dataset.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
# from pgm_dataset.learning.scores import BIC
# from pgm_dataset.learning.algorithms import GreedyHillClimbing
# from pgm_dataset.models import GaussianNetwork
# import util_test

# SIZE = 10000
# df = util_test.generate_normal_data(SIZE)

# def test_create():
#     gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
#     bic = BIC(df)

#     blacklist_arcs = []
#     whitelist_arcs = []

#     # arcs = ArcOperatorSet(gbn, bic, blacklist_arcs, whitelist_arcs, local_score?, 4)
#     arcs = ArcOperatorSet(gbn, bic)

#     # whitelist_type = []
#     # node_type = ChangeNodeTypeSet(gbn, bic, whitelist_type, local_score?)
#     node_type = ChangeNodeTypeSet(gbn, bic)

#     pool = OperatorPool(gbn, bic, [arcs, node_type])

#     # pool = OperatorPool(model)

#     hc = GreedyHillClimbing()
#     hc.estimate(df, pool, 0, 0, gbn)