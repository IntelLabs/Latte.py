class Task:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self):
        self.fn(*self.args)
