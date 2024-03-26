import time
import signal
import pickle
from pyeda.boolalg.bdd import BDDVariable, BDDNODEZERO, BDDNODEONE, bddvar, expr2bdd

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