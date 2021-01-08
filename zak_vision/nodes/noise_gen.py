import numpy as np

from zak_vision.nodes.base_nodes import BaseNode, Edge


class NoiseGen(BaseNode):
    # TODO: do the same thing, but sphere is not centered around the origin
    def __init__(self, output: Edge, params, config):
        super().__init__()
        self.output = output

        self.dim_noise = config['dim_noise']
        self.params = params

        self.pos = None

    def setup(self):
        self.pos = np.random.randn(12, self.dim_noise)
        self.pos = self.make_unit(self.pos)

    @staticmethod
    def make_unit(v):
        norm = np.linalg.norm(v, axis=1)
        return v / norm[:, np.newaxis]

    def task(self):
        w = np.zeros((13, 512))
        chroma = np.frombuffer(self.params['chords_chroma'], dtype='float32')
        chroma = self.pos * chroma[:, np.newaxis] * 100.
        noise = np.random.randn(1, self.dim_noise)
        chroma += noise * self.params['chords_dissonance']
        drums_onset = self.params['drums_onset'].value * np.random.randn(512) * 5.

        w[-1] = drums_onset
        w[:-1] = chroma
        self.write(self.output, w)
