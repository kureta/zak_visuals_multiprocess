import threading
from types import FunctionType

from pythonosc import dispatcher, osc_server, udp_client


class OSCServer(threading.Thread):
    on_force: FunctionType
    on_radius: FunctionType
    on_speed: FunctionType
    on_unknown_message: FunctionType

    def __init__(self, params: dict):
        print('server start')
        super().__init__()
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        ip = '172.16.31.191'
        port = 8000
        self.client = udp_client.SimpleUDPClient(ip, port)

        self.params = params

        self.dispatcher.map('/controls/force', self.on_force)
        self.dispatcher.map('/controls/radius', self.on_radius)
        self.dispatcher.map('/controls/speed', self.on_speed)
        self.dispatcher.set_default_handler(self.on_unknown_message)

    def on_unknown_message(self, addr, *values):
        # print(f'addr: {addr}', f'values: {values}')
        self.client.send_message(addr, values)

    def on_force(self, addr, value):
        print(addr, value)
        self.params['force'].value = value

    def on_radius(self, addr, value):
        print(addr, value)
        self.params['radius'].value = value

    def on_speed(self, addr, value):
        print(addr, value)
        self.params['speed'].value = value

    def run(self):
        self.server.serve_forever()

    def join(self, **kwargs):
        self.server.shutdown()
        super(OSCServer, self).join(**kwargs)
