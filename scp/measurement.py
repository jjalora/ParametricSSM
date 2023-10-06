import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import lil_matrix, vstack

# TODO: Currently only support observing q
class linearModel:
    """ Assumes state vector x = [v; q] and q = [n1.x; n1.y; n1.z; ...]. 
    This class builds a measurement model y = Cx, callable as y = class.evaluate(x) 

    Inputs:
        nodes: list of integer node numbers starting from 0
        num_nodes: total number of nodes (158 for finger)
        pos: True (default) to output position q of each node in nodes
        vel: True (default) to output velocity v of each node in nodes

    y = class.evaluate(x) where x = [v; q] outputs the measurement with 
    [q1;v1;q2;v2;...] format.

    """

    def __init__(self, nodes, num_nodes, spatial_dim=1):
        self.build_C_matrix(nodes, num_nodes, spatial_dim)
        self.num_nodes = num_nodes
        # self.qv = qv

    def build_C_matrix(self, nodes, num_nodes, spatial_dim):
        Cq = buildCq(nodes, num_nodes, spatial_dim)
        self.C = Cq
    def evaluate(self, x, qv=False):
        return self.C @ x

class MeasurementModel(linearModel):
    def __init__(self, nodes, num_nodes, mu_q=None, S_q=None, spatial_dim=1):
        super().__init__(nodes, num_nodes, spatial_dim=spatial_dim)

        pos_dim = self.C.shape[0]

        if mu_q is None:
            mu_q = np.zeros(pos_dim)
        if S_q is None:
            S_q = np.zeros((pos_dim, pos_dim))

        self.mean = mu_q
        self.covariance = S_q

        assert self.mean.shape[0] == self.C.shape[0]
        assert self.covariance.shape[0] == self.C.shape[0] and self.covariance.shape[1] == self.C.shape[0]

    def evaluate(self, x):
        z = self.C @ x + np.random.multivariate_normal(mean=self.mean, cov=self.covariance)
        return z


# def buildCv(nodes, num_nodes, spatial_dim):
#     Cv = lil_matrix((3 * len(nodes), 6 * num_nodes))
#     # Format of x: [vx_0, vy_0, vz_0, (up to num_nodes), qx_0, qy_0, qz_0,...]
#     for (i, node) in enumerate(nodes):
#         Cq[3 * i, 3 * num_nodes + 3 * node] = 1.
#         Cq[3 * i + 1, 3 * num_nodes + 3 * node + 1] = 1.
#         Cq[3 * i + 2, 3 * num_nodes + 3 * node + 2] = 1.
#     return Cq


def buildCq(nodes, num_nodes, spatial_dim):
    Cq = np.zeros((spatial_dim * len(nodes), spatial_dim * num_nodes))
    for (i, node) in enumerate(nodes):
        for j in range(spatial_dim):
            Cq[spatial_dim * i + j, spatial_dim * node + j] = 1.
    return Cq