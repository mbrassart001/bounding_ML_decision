import time
import signal

def handler(sig, frame):
    raise Exception("Function takes too much time")

def timelimit(maxtime=100):
    def inner(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(maxtime)
            try:
                res = func(*args, **kwargs)
            except Exception as exc:
                print(exc)
            else:
                signal.alarm(-1)
                return res
        return wrapper
    return inner

def timecounter(message=None):
    def inner(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            deltatime = (time.time() - start)*1000
            if message:
                try:
                    print(message.format(deltatime))
                except Exception:
                    print(f"Done in {deltatime}ms")
            elif message == False:
                return res, deltatime
            else:
                print(f"Done in {deltatime}ms")
            return res
        return wrapper
    return inner