import multiprocessing as mp


class Edge:
    def __init__(self):
        self.__q = mp.Queue(maxsize=1)

    def close(self):
        self.__q.close()

    def read(self):
        return self.__q.get(block=True, timeout=1 / 30)

    def write(self, value):
        self.__q.put(value, block=True, timeout=1 / 30)


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
