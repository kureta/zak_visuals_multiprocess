from fractions import Fraction
from time import sleep

from gstreamer import Gst, GstApp, GstContext, GstPipeline, GstVideo, utils  # noqa

from zak_vision.nodes.base_nodes import BaseNode, Edge


def fraction_to_str(fraction):
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


VIDEO_FORMAT = 'RGB'
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)


class Streamer(BaseNode):
    def __init__(self, image: Edge, config):
        super().__init__()
        self.image = image
        self.context = None

        fps = Fraction(config['fps'])
        width = config['width']
        height = config['height']

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

    def setup(self):
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
        self.stream_frame(images)

    def teardown(self):
        # emit <end-of-stream> event
        self.appsrc.emit("end-of-stream")
        while not self.pipeline.is_done:
            sleep(.05)
        self.pipeline.shutdown()
        self.context.shutdown()
