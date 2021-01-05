import time
from contextlib import contextmanager

from torch import multiprocessing as mp


@contextmanager
def wait(frame_duration: float):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        wait_duration = frame_duration - (t1 - t0)
        if frame_duration > 0 > wait_duration:
            print(f'WARNING: computation takes more than frame duration {wait_duration:.4f}')
        time.sleep(max(wait_duration, 0))


class Edge:
    def __init__(self):
        self.__q = mp.Queue(maxsize=1)

    def close(self):
        self.__q.close()

    def read(self):
        return self.__q.get()

    def write(self, value):
        self.__q.put(value)


class BaseNode(mp.Process):
    def __init__(self):
        super().__init__()
        self.exit = mp.Event()
        self.pause = mp.Event()

    def run(self):
        self.setup()
        while not self.exit.is_set():
            if self.pause.is_set():
                self.pause.wait()
            self.task()
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
