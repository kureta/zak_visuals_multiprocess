import struct
import threading

from pythonosc import dispatcher, osc_server, udp_client


class OSCServer(threading.Thread):
    def __init__(self, params: dict):
        print('server start')
        super().__init__()
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        ip = '172.16.31.191'
        port = 8000
        self.client = udp_client.SimpleUDPClient(ip, port)

        self.params = params
        self.setup_listeners()

    # noinspection PyTypeChecker
    def setup_listeners(self):
        self.dispatcher.map('/chords/amp', self.on_chords_amp)
        self.dispatcher.map('/chords/chroma', self.on_chords_chroma)
        self.dispatcher.map('/chords/dissonance', self.on_chords_dissonance)
        self.dispatcher.map('/bass/amp', self.on_bass_amp)
        self.dispatcher.map('/bass/has_pitch', self.on_bass_has_pitch)
        self.dispatcher.map('/bass/pitch', self.on_bass_pitch)
        self.dispatcher.map('/drums/amp', self.on_drums_amp)
        self.dispatcher.map('/drums/onset', self.on_drums_onset)
        self.dispatcher.map('/drums/centroid', self.on_drums_centroid)
        self.dispatcher.set_default_handler(self.on_unknown_message)

    def on_unknown_message(self, addr, *values):
        print(f'Unknown message: addr={addr}', f'values={values}')
        self.client.send_message(addr, values)

    def on_chords_amp(self, _addr, value):
        self.params['chords_amp'].value = value

    def on_chords_chroma(self, _addr, *value):
        struct.pack_into('ffffffffffff', self.params[f'chords_chroma'], 0, *value)

    def on_chords_dissonance(self, _addr, value):
        self.params['chords_dissonance'].value = value

    def on_bass_amp(self, _addr, value):
        self.params['bass_amp'].value = value

    def on_bass_has_pitch(self, _addr, value):
        self.params['bass_has_pitch'].value = value

    def on_bass_pitch(self, _addr, value):
        self.params['bass_pitch'].value = value

    def on_drums_amp(self, _addr, value):
        self.params['drums_amp'].value = value

    def on_drums_onset(self, _addr, value):
        self.params['drums_onset'].value = value

    def on_drums_centroid(self, _addr, value):
        self.params['drums_centroid'].value = value

    def run(self):
        self.server.serve_forever()

    def join(self, **kwargs):
        self.server.shutdown()
        super(OSCServer, self).join(**kwargs)
