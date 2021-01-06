import ctypes
import signal
from multiprocessing import Event, Value, set_start_method

from zak_vision.nodes import Generator, Noise, OSCServer, Streamer
from zak_vision.nodes.base_nodes import Edge

config = {
    'width': 1024,
    'height': 1024,
    'dim_noise': 512,
    'batch_size': 1,
    'network': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl',
    'fps': 30,
}


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
        self.noise_gen = Noise(self.noise, params, config)
        self.generator = Generator(self.noise, self.images, config)
        self.streamer = Streamer(self.images, config)
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
