import sys, inspect

from pyeda.boolalg.bdd import _bdd, BDDNODEZERO, BDDNODEONE
from exp.expbase import ExpBase

class Exp1(ExpBase):
    def robdd_remove_methods(self, remove_inputs):
        up_method = lambda robdd: robdd.compose({k: _bdd(BDDNODEZERO) for k in remove_inputs})
        down_method = lambda robdd: robdd.compose({k: _bdd(BDDNODEONE) for k in remove_inputs})
        return {"up": up_method, "down": down_method}

class Exp2(ExpBase):
    def robdd_remove_methods(self, remove_inputs):
        up_method = lambda robdd: robdd.consensus(remove_inputs)
        down_method = lambda robdd: robdd.smoothing(remove_inputs)
        return {"up": up_method, "down": down_method}

def main(expname, filename):
    classes = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))
    expclass = classes.get(expname.capitalize(), None)
    if expclass is None:
        raise ValueError(f"{expname.capitalize()} is not a valid experience name")
    exp = expclass(filename)
    return exp.main(verbose=True)
