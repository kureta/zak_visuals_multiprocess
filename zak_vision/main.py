import ctypes
import signal
from cmath import isclose
from fractions import Fraction
from multiprocessing import Event, Value, set_start_method
from time import sleep

import nltk
import numpy as np
import torch
from gstreamer import Gst, GstApp, GstContext, GstPipeline, GstVideo, utils  # noqa
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names)

from zak_vision.base_nodes import BaseNode, Edge, wait
from zak_vision.osc import OSCServer

nltk.data.path.append('/home/kureta/Documents/ML Files/nltk_data')

WIDTH = HEIGHT = 512
NUM_LABELS = 1000
DIM_NOISE = 128
BATCH_SIZE = 12
# TODO: Queue makes generator stop while it's sending the current batch

DONE = np.ones(1, dtype='uint8') * DIM_NOISE


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


class Generator(BaseNode):
    def __init__(self, noise: Edge, image: Edge):
        super().__init__()
        self.noise = noise
        self.image = image

        self.model = self.class_vector = self.position = None
        self.truncation = 1.

    def setup(self):
        self.model = BigGAN.from_pretrained(f'/home/kureta/Documents/repos/personal/zak_vision/biggan-deep-{HEIGHT}')
        self.position = torch.zeros(BATCH_SIZE, DIM_NOISE).to('cuda')

        # Prepare a input
        class_vector = one_hot_from_names(['tiger'], batch_size=BATCH_SIZE)

        # All in tensors
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        self.class_vector = class_vector.to('cuda')
        self.model.to('cuda')

    def task(self):
        for idx in range(BATCH_SIZE):
            self.position[idx] = self.noise.read()

        # Generate an image
        with torch.no_grad():
            output = self.model(self.position, self.class_vector, self.truncation)

        output = output.permute(0, 2, 3, 1)
        output = (output + 1) / 2
        output *= 255
        output = torch.clip(output, 0, 255)
        output = output.cpu().numpy().astype('uint8')

        self.image.write(output)

    def teardown(self):
        self.image.write(DONE)
        self.image.close()


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
        with torch.no_grad():
            self.pos: torch.Tensor = torch.randn(DIM_NOISE).to('cuda')
            self.pos = self.make_unit(self.pos)
            self.pos *= self.radius.value

            self.vel: torch.Tensor = torch.randn(DIM_NOISE).to('cuda')
            self.vel = self.make_orthogonal_to(self.vel, self.pos)
            self.vel = self.make_unit(self.vel)
            self.vel *= self.speed.value

            self.acc: torch.Tensor = torch.randn(DIM_NOISE).to('cuda')
            self.acc = self.make_orthogonal_to(self.acc, self.vel)
            self.acc = self.make_orthogonal_to(self.acc, self.pos)
            self.acc = self.make_unit(self.acc)
            self.acc *= self.force.value

    @staticmethod
    def make_unit(v: torch.Tensor) -> torch.Tensor:
        norm = v.norm(p=2)
        if isclose(norm, 0.):
            return v
        return v / norm

    def make_orthogonal_to(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        v_r = torch.dot(v1, v2) * self.make_unit(v2)
        return v1 - v_r

    def apply_force(self):
        self.acc.normal_()
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
        with torch.no_grad():
            self.apply_force()
        self.output.write(self.pos)


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
    "v4l2sink device=/dev/video0 sync=false"
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

    def task(self):
        with wait(1 / 30):
            try:
                image = self.image.read()
                if np.all(image == DONE):
                    self.exit.set()
                    return
                gst_buffer = utils.ndarray_to_gst_buffer(image)

                # set pts and duration to be able to record video, calculate fps
                self.pts += self.duration  # Increase pts by duration
                gst_buffer.pts = self.pts
                gst_buffer.duration = self.duration

                # emit <push-buffer> event with Gst.Buffer
                self.appsrc.emit("push-buffer", gst_buffer)
            except Exception as e:
                print("Error: ", e)

    def teardown(self):
        # emit <end-of-stream> event
        self.appsrc.emit("end-of-stream")
        while not self.pipeline.is_done:
            sleep(.1)
        self.pipeline.shutdown()
        self.context.shutdown()
        self.image.close()


class Queue(BaseNode):
    # TODO: make queue more genric
    #       It should take any number of items at a time and send them in any size batches, or one by one
    def __init__(self, images: Edge, image: Edge, wait_duration=1 / 24):
        super().__init__()
        self.images = images
        self.image = image
        self.wait_duration = wait_duration

    def task(self):
        images = self.images.read()
        if np.all(images == DONE):
            self.exit.set()
            return
        for img in images:
            self.image.write(img)

    def teardown(self):
        self.image.write(DONE)
        self.images.close()
        self.image.close()


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

        self.image = Edge()
        self.images = Edge()
        self.noise = Edge()
        self.noise_gen = Noise(self.noise, params)
        self.generator = Generator(self.noise, self.images)
        self.queue = Queue(self.images, self.image, wait_duration=1 / (FPS + 2))
        self.streamer = Streamer(self.image)
        self.osc = OSCServer(params)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.noise_gen.start()
        self.generator.start()
        self.queue.start()
        self.streamer.start()
        self.osc.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.noise_gen.exit.set()
        self.noise_gen.join()
        self.generator.join()
        self.queue.join()
        self.streamer.join()
        self.osc.join()

        self.image.close()
        self.images.close()
        self.noise.close()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
