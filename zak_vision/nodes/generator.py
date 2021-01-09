import pickle

import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.nodes.base_nodes import BaseNode, Edge


class Generator(BaseNode):
    def __init__(self, noise: Edge, image: Edge, config, params):
        super().__init__()
        self.noise = noise
        self.image = image

        self.dim_noise = config['dim_noise']
        self.network = config['network']
        self.params = params

        self.buffer = np.zeros((1, 13, config['dim_noise']))

        self.Gs = self.Gs_kwargs = self.noise_vars = self.noise_values = self.label = self.latents = self.dlatents = None

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
        self.noise_values = [np.random.randn(*var.shape.as_list()) for var in self.noise_vars]
        tflib.set_vars({var: self.noise_values[idx] for idx, var in enumerate(self.noise_vars)})

        self.latents = np.random.randn(1, self.dim_noise)
        self.dlatents = self.Gs.components.mapping.run(self.latents, None)

    def task(self):
        self.buffer = self.read(self.noise)
        if self.buffer is None:
            return
        # drums_amp = self.params['drums_amp'].value
        # drums_onset = self.params['drums_onset'].value
        # drums_centroid = self.params['drums_centroid'].value
        # tflib.set_vars(
        #     {self.noise_vars[idx]: 50 * drums_onset * drums_amp * (1 - drums_centroid) * self.noise_values[idx] for idx in
        #      range(9)})
        # tflib.set_vars({self.noise_vars[idx]: 50 * drums_onset * drums_amp * drums_centroid * self.noise_values[idx] for idx in
        #                 range(9, 17)})
        self.dlatents = self.Gs.components.mapping.run(self.buffer[np.newaxis, :], None)

        images = self.Gs.components.synthesis.run(self.dlatents, **self.Gs_kwargs)
        self.write(self.image, images)
