import pickle

import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.nodes.base_nodes import BaseNode, Edge


def random_orthonormal(n, m=512):
    h = np.random.randn(n, m)
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    mat = u @ vh

    return mat


class Generator(BaseNode):
    def __init__(self, image: Edge, config, params):
        super().__init__()
        self.image = image

        self.dim_noise = config['dim_noise']
        self.network = config['network']
        self.params = params

        self.buffer = np.zeros((1, 13, config['dim_noise']))

        self.Gs = self.Gs_kwargs = self.noise_vars = self.noise_values = None
        self.label = self.latents = self.dlatents = self.chroma = None

    def setup(self):
        tflib.init_tf()
        with dnnlib.util.open_url(self.network) as fp:
            _G, _D, self.Gs = pickle.load(fp)

        self.Gs_kwargs = {
            'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            'randomize_noise': False,
            'truncation_psi': 0,
        }

        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        self.noise_values = [np.random.randn(*var.shape.as_list()) for var in self.noise_vars]
        tflib.set_vars({var: self.noise_values[idx] for idx, var in enumerate(self.noise_vars)})

        self.latents = np.random.randn(1, self.dim_noise)
        self.dlatents = self.Gs.components.mapping.run(self.latents, None)
        self.chroma = random_orthonormal(12, self.dim_noise)

    def task(self):
        chords_chroma = np.frombuffer(self.params['chords_chroma'], dtype='float32')
        chords_chroma = np.sum(self.chroma * chords_chroma[:, np.newaxis], axis=0)

        self.dlatents = self.Gs.components.mapping.run(chords_chroma[np.newaxis, :], None)
        for i in range(18):
            self.dlatents[0, i, :] += chords_chroma

        images = self.Gs.components.synthesis.run(self.dlatents, **self.Gs_kwargs)
        self.write(self.image, images)
