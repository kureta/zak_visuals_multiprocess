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
        chroma = np.frombuffer(self.params['chords_chroma'], dtype='float32')
        pos = self.pos * chroma[:, np.newaxis] * 100.
        shit = np.zeros((13, 512))
        shit[0] = self.params['drums_onset'].value * np.random.randn(512) * 100.
        shit[1:] = pos
        self.write(self.output, shit)
