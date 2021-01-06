import numpy as np

from zak_vision.nodes.base_nodes import BaseNode, Edge


class Noise(BaseNode):
    # TODO: do the same thing, but sphere is not centered around the origin
    def __init__(self, output: Edge, params, config):
        super().__init__()
        self.output = output

        self.dim_noise = config['dim_noise']

        self.force = params['force']
        self.speed = params['speed']
        self.radius = params['radius']

        self.pos = None
        self.vel = None
        self.acc = None

    def setup(self):
        self.pos = np.random.randn(self.dim_noise)
        self.pos = self.make_unit(self.pos)
        self.pos *= self.radius.value

        self.vel = np.random.randn(self.dim_noise)
        self.vel = self.make_orthogonal_to(self.vel, self.pos)
        self.vel = self.make_unit(self.vel)
        self.vel *= self.speed.value

        self.acc = np.random.randn(self.dim_noise)
        self.acc = self.make_orthogonal_to(self.acc, self.vel)
        self.acc = self.make_orthogonal_to(self.acc, self.pos)
        self.acc = self.make_unit(self.acc)
        self.acc *= self.force.value

    @staticmethod
    def make_unit(v):
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0.):
            return v
        return v / norm

    def make_orthogonal_to(self, v1, v2):
        v_r = np.dot(v1, v2) * self.make_unit(v2)
        return v1 - v_r

    def apply_force(self):
        self.acc = np.random.randn(*self.acc.shape)
        # self.acc = self.make_orthogonal_to(self.acc, self.vel)
        # self.acc = self.make_orthogonal_to(self.acc, self.pos)
        # self.acc = self.make_unit(self.acc)
        self.acc *= self.force.value

        self.vel += self.acc
        # self.vel = self.make_orthogonal_to(self.vel, self.pos)
        self.vel = self.make_unit(self.vel)
        self.vel *= self.speed.value

        self.pos += self.vel
        self.pos = self.make_unit(self.pos)
        self.pos *= self.radius.value

    def task(self):
        self.apply_force()
        self.write(self.output, self.pos)
