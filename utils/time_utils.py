import time


class Timer:

    def __init__(self):
        self.start_time = time.time()

    def tick(self, msg=""):
        end_time = time.time()
        print("Tick:{}:{}".format(msg, end_time-self.start_time))
        self.start_time = time.time()
