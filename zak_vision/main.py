import ctypes
import signal
from multiprocessing import Event, set_start_method
from multiprocessing.sharedctypes import RawArray, RawValue  # noqa

from zak_vision.nodes import Generator, OSCServer

config = {
    'width': 1024,
    'height': 1024,
    'dim_noise': 512,
    'network': '/home/kureta/Documents/stylegan2-pretrained/metfaces.pkl',
    'fps': 30,
}


class App:
    def __init__(self):
        set_start_method('spawn', force=True)

        params = {
            'chords_amp': RawValue(ctypes.c_float),
            'chords_chroma': RawArray(ctypes.c_float, 12 * [0.]),
            'chords_dissonance': RawValue(ctypes.c_float),
            'bass_amp': RawValue(ctypes.c_float),
            'bass_pitch': RawValue(ctypes.c_float),
            'bass_has_pitch': RawValue(ctypes.c_float),
            'drums_amp': RawValue(ctypes.c_float),
            'drums_onset': RawValue(ctypes.c_float),
            'drums_centroid': RawValue(ctypes.c_float),
        }

        params['chords_amp'].value = 0.
        params['chords_dissonance'].value = 0.
        params['bass_amp'].value = 0.
        params['bass_pitch'].value = 0.
        params['bass_has_pitch'].value = 0.
        params['drums_amp'].value = 0.
        params['drums_onset'].value = 0.
        params['drums_centroid'].value = 0.

        self.generator = Generator(config, params)
        self.osc = OSCServer(params)

        self.exit = Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.generator.start()
        self.osc.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()

    def exit_handler(self):
        self.generator.join()
        self.osc.join()

        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
