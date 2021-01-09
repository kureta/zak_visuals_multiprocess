import numpy as np

from zak_vision.nodes.base_nodes import BaseNode, Edge


def random_orthonormal(n, m=512):
    H = np.random.randn(n, m)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat = u @ vh

    return mat


class NoiseGen(BaseNode):
    # TODO: do the same thing, but sphere is not centered around the origin
    def __init__(self, output: Edge, params, config):
        super().__init__()
        self.output = output

        self.dim_noise = config['dim_noise']
        self.params = params

        self.chroma = None

    def setup(self):
        # TODO: add randomizable offset
        # TODO: fix croma and amp scaling
        self.chroma = random_orthonormal(12, self.dim_noise)

    @staticmethod
    def make_unit(v):
        norm = np.linalg.norm(v, axis=1)
        return v / norm[:, np.newaxis]

    def task(self):
        chords_chroma = np.frombuffer(self.params['chords_chroma'], dtype='float32')
        chords_chroma = np.sum(self.chroma * chords_chroma[:, np.newaxis], axis=0)
        chords_amp = self.params['chords_amp'].value

        chords_chroma *= chords_amp * 50.

        self.write(self.output, chords_chroma)
