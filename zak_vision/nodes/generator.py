import pickle
from fractions import Fraction
from time import sleep

import numpy as np
from gstreamer import Gst, GstApp, GstContext, GstPipeline, GstVideo, utils  # noqa

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.nodes.base_nodes import BaseNode


def random_orthonormal(n, m=512):
    h = np.random.randn(n, m)
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    mat = u @ vh

    return mat


def fraction_to_str(fraction):
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


VIDEO_FORMAT = 'RGB'
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)


class Generator(BaseNode):
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    # noinspection PyAttributeOutsideInit
    def setup(self):
        fps = Fraction(self.config['fps'])
        width = self.config['width']
        height = self.config['height']

        fps_str = fraction_to_str(fps)
        self.caps = f'video/x-raw,format={VIDEO_FORMAT},width={width},height={height},framerate={fps_str}'

        # Converts list of plugins to gst-launch string
        # ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
        self.default_pipeline = utils.to_gst_string([
            f'appsrc caps={self.caps}',
            'videoconvert',
            'v4l2sink device=/dev/video0 sync=false'
        ])

        self.duration = 10 ** 9 / (fps.numerator / fps.denominator)
        self.appsrc = self.pts = self.pipeline = None

        self.context = GstContext()
        self.context.startup()
        self.pipeline = GstPipeline(self.default_pipeline)

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
            self.appsrc.set_caps(Gst.Caps.from_string(self.caps))

        # override on_pipeline_init to set specific properties before launching pipeline
        self.pipeline._on_pipeline_init = on_pipeline_init.__get__(self.pipeline)  # noqa

        try:
            self.pipeline.startup()
            self.appsrc = self.pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc

            self.pts = 0  # buffers presentation timestamp
        except Exception as e:
            print("Error: ", e)

        self.dim_noise = self.config['dim_noise']
        self.network = self.config['network']

        self.buffer = np.zeros((1, 13, self.config['dim_noise']))

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
