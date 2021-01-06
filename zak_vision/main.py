import ctypes
import pickle
import signal
from fractions import Fraction
from multiprocessing import Event, Value, set_start_method
from time import sleep

import numpy as np
from gstreamer import Gst, GstApp, GstContext, GstPipeline, GstVideo, utils  # noqa

import dnnlib
import dnnlib.tflib as tflib
from zak_vision.base_nodes import BaseNode, Edge
from zak_vision.osc import OSCServer

WIDTH = HEIGHT = 1024
NUM_LABELS = 1000
DIM_NOISE = 512
BATCH_SIZE = 1
NET = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl'


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


class Generator(BaseNode):
    def __init__(self, noise: Edge, image: Edge):
        super().__init__()
        self.noise = noise
        self.image = image
        self.buffer = np.zeros((BATCH_SIZE, DIM_NOISE))

        self.Gs = self.Gs_kwargs = self.noise_vars = self.label = self.latents = self.dlatents = None

    def setup(self):
        tflib.init_tf()
        with dnnlib.util.open_url(NET) as fp:
            _G, _D, self.Gs = pickle.load(fp)

        self.Gs_kwargs = {
            'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            'randomize_noise': False,
            'truncation_psi': 1.0,
        }

        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in self.noise_vars})

        latent = np.random.randn(DIM_NOISE)
        self.latents = np.stack([latent for _ in range(BATCH_SIZE)])
        self.dlatents = self.Gs.components.mapping.run(self.latents, None)

    def task(self):
        for idx in range(BATCH_SIZE):
            buffer = self.read(self.noise)
            if buffer is None:
                return
            self.buffer[idx] = buffer

        for i in range(self.dlatents.shape[1]):
            self.dlatents[:, i] = self.buffer
        images = self.Gs.components.synthesis.run(self.dlatents, **self.Gs_kwargs)
        self.write(self.image, images)


class Noise(BaseNode):
    # TODO: do the same thing, but sphere is not centered around the origin
    def __init__(self, output: Edge, params):
        super().__init__()
        self.output = output

        self.force = params['force']
        self.speed = params['speed']
        self.radius = params['radius']

        self.pos = None
        self.vel = None
        self.acc = None

    def setup(self):
        self.pos = np.random.randn(DIM_NOISE)
        self.pos = self.make_unit(self.pos)
        self.pos *= self.radius.value

        self.vel = np.random.randn(DIM_NOISE)
        self.vel = self.make_orthogonal_to(self.vel, self.pos)
        self.vel = self.make_unit(self.vel)
        self.vel *= self.speed.value

        self.acc = np.random.randn(DIM_NOISE)
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


VIDEO_FORMAT = "RGB"
FPS = Fraction(30)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)
FPS_STR = fraction_to_str(FPS)
CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())
CHANNELS = utils.get_num_channels(GST_VIDEO_FORMAT)
DTYPE = utils.get_np_dtype(GST_VIDEO_FORMAT)

# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc caps={CAPS}".format(**locals()),
    "videoconvert",
    "v4l2sink device=/dev/video0 sync=true"
])


class Streamer(BaseNode):
    def __init__(self, image: Edge):
        super().__init__()
        self.image = image
        self.context = None

        self.appsrc = self.pts = self.duration = self.pipeline = None

    def setup(self):
        self.context = GstContext()
        self.context.startup()
        self.pipeline = GstPipeline(DEFAULT_PIPELINE)

        def on_pipeline_init(other_self):
            """Setup AppSrc element"""
            self.appsrc = other_self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc

            # instructs appsrc that we will be dealing with timed buffer
            self.appsrc.set_property("format", Gst.Format.TIME)

            # instructs appsrc to block pushing buffers until ones in queue are preprocessed
            # allows to avoid huge queue internal queue size in appsrc
            self.appsrc.set_property("block", True)

            # set input format (caps)
            self.appsrc.set_caps(Gst.Caps.from_string(CAPS))

        # override on_pipeline_init to set specific properties before launching pipeline
        self.pipeline._on_pipeline_init = on_pipeline_init.__get__(self.pipeline)  # noqa

        try:
            self.pipeline.startup()
            self.appsrc = self.pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc

            self.pts = 0  # buffers presentation timestamp
            self.duration = 10 ** 9 / (FPS.numerator / FPS.denominator)  # frame duration
        except Exception as e:
            print("Error: ", e)

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
        images = self.read(self.image)
        if images is None:
            return
        for im in images:
            self.stream_frame(im)

    def teardown(self):
        # emit <end-of-stream> event
        self.appsrc.emit("end-of-stream")
        while not self.pipeline.is_done:
            sleep(.05)
        self.pipeline.shutdown()
        self.context.shutdown()


class App:
    def __init__(self):
        set_start_method('spawn', force=True)

        params = {
            'force': Value(ctypes.c_float, lock=False),
            'radius': Value(ctypes.c_float, lock=False),
            'speed': Value(ctypes.c_float, lock=False),
        }

        params['force'].value = 0.5
        params['radius'].value = 8.
        params['speed'].value = 0.95

        self.images = Edge()
        self.noise = Edge()
        self.noise_gen = Noise(self.noise, params)
        self.generator = Generator(self.noise, self.images)
        self.streamer = Streamer(self.images)
        self.osc = OSCServer(params)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.noise_gen.start()
        self.generator.start()
        self.streamer.start()
        self.osc.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.noise_gen.join()
        self.generator.join()
        self.streamer.join()
        self.osc.join()

        self.images.close()
        self.noise.close()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
