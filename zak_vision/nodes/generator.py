import pickle

import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.nodes.base_nodes import BaseNode, Edge


class Generator(BaseNode):
    def __init__(self, noise: Edge, image: Edge, config):
        super().__init__()
        self.noise = noise
        self.image = image

        self.dim_noise = config['dim_noise']
        self.batch_size = config['batch_size']
        self.network = config['network']

        self.buffer = np.zeros((config['batch_size'], 13, config['dim_noise']))

        self.Gs = self.Gs_kwargs = self.noise_vars = self.label = self.latents = self.dlatents = None

    def setup(self):
        tflib.init_tf()
        with dnnlib.util.open_url(self.network) as fp:
            _G, _D, self.Gs = pickle.load(fp)

        self.Gs_kwargs = {
            'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            'randomize_noise': False,
            'truncation_psi': 1.0,
        }

        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in self.noise_vars})

        latent = np.random.randn(self.dim_noise)
        self.latents = np.stack([latent for _ in range(self.batch_size)])
        self.dlatents = self.Gs.components.mapping.run(self.latents, None)

    def task(self):
        for idx in range(self.batch_size):
            buffer = self.read(self.noise)
            if buffer is None:
                return
            self.buffer[idx] = buffer

        for i in range(self.dlatents.shape[1]):
            self.dlatents[:, i] = self.buffer[:, i % self.buffer.shape[1]]
        images = self.Gs.components.synthesis.run(self.dlatents, **self.Gs_kwargs)
        self.write(self.image, images)
