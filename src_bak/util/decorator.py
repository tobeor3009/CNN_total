from time import time

class TimeSpend:

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        start_time = time()
        self.function(*args, **kwargs)
        print(time() - start_time)