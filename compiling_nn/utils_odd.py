import time
import signal
import threading
import functools
import pickle
from pyeda.boolalg.bdd import BDDVariable, BDDNODEZERO, BDDNODEONE, bddvar, expr2bdd

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Function took too long!")

def timelimit(seconds=100):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGINT, _timeout_handler)

            def watchdog():
                time.sleep(seconds)
                signal.raise_signal(signal.SIGINT)

            t = threading.Thread(target=watchdog)
            t.daemon = True
            t.start()

            try:
                return func(*args, **kwargs)
            except TimeoutException as e:
                raise e
            finally:
                signal.signal(signal.SIGINT, signal.default_int_handler)
        return wrapper
    return decorator

def timecounter(message=None):
    def decorator(func):
        @functools.wraps(func)
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
    return decorator

class BDDPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if obj is BDDNODEZERO:
            return "bddnodezero"
        elif obj is BDDNODEONE:
            return "bddnodeone"
        elif isinstance(obj, BDDVariable):
            return f"bddvar {obj.name}"
        else:
            return None

class BDDUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if pid == "bddnodezero":
            return BDDNODEZERO
        elif pid == "bddnodeone":
            return BDDNODEONE
        subpid, name = pid.split(" ")
        if subpid == "bddvar":
            return bddvar(name)
        else:
            raise pickle.UnpicklingError("unsupported persistent object")

def pickle_bdd(bdd_obj, filepath):
    with open(filepath, 'wb') as f:
        BDDPickler(f).dump(bdd_obj)

def unpickle_bdd(filepath):
    with open(filepath, 'rb') as f:
        unpickled_bdd = BDDUnpickler(f).load()
    return expr2bdd(unpickled_bdd)