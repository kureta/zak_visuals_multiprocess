import pickle
from fractions import Fraction
from time import sleep

import numpy as np
from gstreamer import Gst, GstApp, GstContext, GstPipeline, GstVideo, utils  # noqa

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.nodes.base_nodes import BaseNode

VIDEO_FORMAT = 'RGB'
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)
FPS = 30
NETWORK = '/home/kureta/Documents/stylegan2-pretrained/metfaces.pkl'


def random_orthonormal(n, m=512):
    h = np.random.randn(n, m).astype('float32')
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    mat = u @ vh

    return mat


def fraction_to_str(fraction):
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


class Generator(BaseNode):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Network attributes
        self.Gs = self.Gs_kwargs = None

        # Latent and noise attributes
        self.noise_values = self.noise_vars = self.latents = None
        self.dlatents = self.chroma = None
        self.origin = self.noise_values2 = None

        # Streamer attributes
        self.duration = self.appsrc = self.pipeline = None
        self.context = self.pts = None

    def setup_network(self):
        tflib.init_tf()
        with dnnlib.util.open_url(NETWORK) as fp:
            _G, _D, self.Gs = pickle.load(fp)
            del _G
            del _D

        self.Gs_kwargs = {
            'output_transform': dict(func=tflib.convert_images_to_uint8,
                                     nchw_to_nhwc=True),
            'randomize_noise': False,
            'truncation_psi': 0,
        }

        dim_noise = self.Gs.input_shape[1]
        width = self.Gs.output_shape[2]
        height = self.Gs.output_shape[3]

        print('Building graph for the first time')
        _ = self.Gs.run(np.zeros((1, dim_noise), dtype='float32'), np.zeros((1, 1), dtype='float32'), **self.Gs_kwargs)

        return dim_noise, width, height

    def setup_latents(self, dim_noise):
        self.origin = np.random.randn(1, dim_noise).astype('float32')
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        self.noise_values = [np.random.randn(*var.shape.as_list()).astype('float32') for var in self.noise_vars]
        self.noise_values2 = [np.random.randn(*var.shape.as_list()).astype('float32') for var in self.noise_vars]
        tflib.set_vars({var: self.noise_values[idx] for idx, var in enumerate(self.noise_vars)})

        self.latents = np.random.randn(1, dim_noise).astype('float32')
        self.dlatents = self.Gs.components.mapping.run(self.latents, None)
        self.chroma = random_orthonormal(12, dim_noise)

    def setup_streamer(self, width, height):
        fps = Fraction(FPS)
        fps_str = fraction_to_str(fps)
        caps = f'video/x-raw,format={VIDEO_FORMAT},width={width},height={height},framerate={fps_str}'

        # Converts list of plugins to gst-launch string
        # ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
        default_pipeline = utils.to_gst_string([
            f'appsrc caps={caps}',
            'videoconvert',
            'v4l2sink device=/dev/video0 sync=false'
        ])

        self.duration = 10 ** 9 / (fps.numerator / fps.denominator)
        self.appsrc = self.pts = self.pipeline = None

        self.context = GstContext()
        self.context.startup()
        self.pipeline = GstPipeline(default_pipeline)

        def on_pipeline_init(other_self):
            """Setup AppSrc element"""
            self.appsrc = other_self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc

            # instructs appsrc that we will be dealing with timed buffer
            self.appsrc.set_property("format", Gst.Format.TIME)

            # instructs appsrc to block pushing buffers until ones in queue are preprocessed
            # allows to avoid huge queue internal queue size in appsrc
            self.appsrc.set_property("block", True)
            self.appsrc.set_property("is-live", True)

            # set input format (caps)
            self.appsrc.set_caps(Gst.Caps.from_string(caps))

        # override on_pipeline_init to set specific properties before launching pipeline
        self.pipeline._on_pipeline_init = on_pipeline_init.__get__(self.pipeline)  # noqa

        try:
            self.pipeline.startup()
            self.appsrc = self.pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc

            self.pts = 0  # buffers presentation timestamp
        except Exception as e:
            print("Error: ", e)

    def setup(self):
        print('Loading network checkpoint...')
        dim_noise, width, height = self.setup_network()
        print('Setting up initial latents and noise...')
        self.setup_latents(dim_noise)
        print('Setting up streamer...')
        self.setup_streamer(width, height)
        print('Ready!')

    def stream_frame(self, image):
        try:
            gst_buffer = utils.ndarray_to_gst_buffer(image)

            # set pts and duration to be able to record video, calculate fps
            self.pts += self.duration  # Increase pts by duration
            gst_buffer.pts = self.pts
            gst_buffer.duration = self.duration

            # emit <push-buffer> event with Gst.Buffer
            self.appsrc.emit("push-buffer", gst_buffer)
        except Exception as e:
            print("Error: ", e)

    def task(self):
        chords_chroma = np.frombuffer(self.params['chords_chroma'], dtype='float32')
        chords_chroma = np.sum(self.chroma * chords_chroma[:, np.newaxis], axis=0)

        # drums_amp = self.params['drums_amp'].value
        # drums_onset = self.params['drums_onset'].value
        # drums_centroid = self.params['drums_centroid'].value
        # val = drums_onset * drums_amp * drums_centroid
        #
        # nv = [val * n1 + (1 - val) * n2 for n1, n2 in zip(self.noise_values, self.noise_values2)]
        # tflib.set_vars({var: nv[idx] for idx, var in enumerate(self.noise_vars)})

        self.dlatents = self.Gs.components.mapping.run(chords_chroma[np.newaxis, :], None)
        for i in range(18):
            self.dlatents[0, i, :] += chords_chroma

        images = self.Gs.components.synthesis.run(self.dlatents, **self.Gs_kwargs)
        self.stream_frame(images)

    def teardown(self):
        # emit <end-of-stream> event
        self.appsrc.emit("end-of-stream")
        while not self.pipeline.is_done:
            sleep(.05)
        self.pipeline.shutdown()
        self.context.shutdown()
