import multiprocessing as mp


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
