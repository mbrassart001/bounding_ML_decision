import time
import pyeda.inter as inter
import pyeda.boolalg as boolalg
from graphviz import Digraph, Source
from IPython.display import SVG, display
from .utils_odd import timelimit, timecounter

infty = float('inf')

class ODD():
    def __init__(self):
        self.root = None
        self.cache = dict()
        self.bdd = None
        self.svg_graph = None
        self.eda_vars = dict() # only for restrict bdd

    def __str__(self):
        return str(boolalg.bdd.bdd2expr(self.root.eda_expr))

    def display_expr(self):
        if self.bdd is not None:
            dot_odd = self.bdd.to_dot()
            display(Source(dot_odd))

    def display_graph(self):
        if not self.svg_graph:
            self.make_graph()
        display(SVG(self.svg_graph))

    def clear_cache(self):
        self.cache.clear()

    def store_in_cache(self, k, node):
        if self.cache.get(k):
            self.cache[k].append(node)
        else:
            self.cache.update({k: [node]})

    def find_in_cache(self, k, value):
        cache_line = self.cache.get(k)
        if cache_line:
            for node in cache_line:
                if value in node.interval:
                    return node
        return None

    def cache_total_node_count(self):
        node_count = 0
        for x in self.cache.values():
            node_count+=x.__len__()
        return node_count

    def rebuild_cache(self, odd):
        self.clear_cache()
        self.cache[0] = [odd]
        has_child = odd.has_child()
        depth = 0
        next_layer = set()
        while has_child:
            layer = self.cache[depth]
            for node in layer:
                for child in node.child.values():
                    next_layer.add(child)

            depth+=1
            self.cache[depth] = list(next_layer)
            next_layer.clear()
            has_child = self.cache[depth][0].has_child()
        
        return self.cache

    def make_graph(self):
        if self.svg_graph is None:
            self._make_graph()
        return self.svg_graph
    
    def _make_graph(self):
        dot = Digraph()
        for cache_line in self.cache.values():
            for node in cache_line:
                dot.node(node.id, f"{node.interval:.2f}")
                for e, child in node.child.items():
                    dot.edge(node.id, child.id, str(e))

        self.svg_graph = dot._repr_image_svg_xml()

    @timecounter(message=False)
    def timed_build_odd_rec(self, weights, threshold):
        return self._build_odd_rec(weights, threshold)

    def build_odd_rec(self, weights, threshold, label=""):
        self.weights = weights
        self.label = label
        self.root = self._build_odd_rec(weights, threshold)
        self.bdd = self.root.eda_expr

    @timelimit(600)
    def _build_odd_rec(self, weights, threshold):
        n = len(weights)
        zero_sink = Node(Interval(-infty, threshold, closed_left=False, closed_right=False))
        zero_sink.eda_expr = 0
        self.zero = zero_sink
        self.store_in_cache(n, zero_sink)

        one_sink = Node(Interval(threshold, infty, closed_right=False))
        one_sink.eda_expr = 1
        self.store_in_cache(n, one_sink)
        self.one = one_sink

        return self.build_sub_odd_rec(0, 0)

    def build_sub_odd_rec(self,  k, v):
        node = Node(Interval(-infty, infty, closed_left=False, closed_right=False))
        eda_var = inter.bddvar(f"{self.label}{k}")
        self.eda_vars[eda_var] = self.svg_graph
        weight = self.weights[k]
        for e in {0, 1}:
            w = e*weight
            v_child = v + w
            child = self.find_in_cache(k+1, v_child)
            if child is None:
                child = self.build_sub_odd_rec(k+1, v_child)
            node.add_child(child, e)
            node.interval.intersect(child.interval-w)
        node.eda_expr = inter.ite(eda_var, node.child[1].eda_expr, node.child[0].eda_expr)
        self.store_in_cache(k, node)
        return node
                

class Node:
    def __init__(self, interval=None):
        self.child = dict()
        self.parent = dict()
        self.interval = interval
        self.id = f"n{id(self)}"
        self.eda_expr = None

    def __str__(self):
        return f"{self.child.__len__()} children | {self.parent.__len__()} parents | {self.interval:.2f}"
    
    def add_child(self, other, label):
        self.child.update({label: other})
        other.parent.update({label: self})

    def pop_child(self, label):
        child = self.child.pop(label)
        child.parent.pop(label)
        return child

    def has_child(self):
        return len(self.child) > 0

class Interval:
    def __init__(self, low, high, closed_left=True, closed_right=True):
        self.low = low
        self.high = high
        self.left = closed_left
        self.right = closed_right
    
    def __str__(self):
        left = "[" if self.left else "("
        right = "]" if self.right else ")"
        return f"{left}{self.low}; {self.high}{right}"

    def __format__(self, __format_spec):
        left = "[" if self.left else "("
        right = "]" if self.right else ")"
        if __format_spec:
            return f"{left}{self.low:{__format_spec}}; {self.high:{__format_spec}}{right}"
        else:
            return f"{left}{self.low}; {self.high}{right}"

    def __add__(self, value):
        return Interval(self.low + value, self.high + value, self.left, self.right)

    def __sub__(self, value):
        return Interval(self.low - value, self.high - value, self.left, self.right)

    def __contains__(self, value):
        down = value >= self.low if self.left else value > self.low
        up = value <= self.high if self.right else value < self.high
        return down & up
    
    def intersect(self, other):
        if self.low < other.low:
            self.low = other.low
            self.left = other.left
        elif self.low == other.low:
            self.left &= other.left
        
        if self.high > other.high:
            self.high = other.high
            self.right = other.right
        elif self.high == other.high:
            self.right &= other.right

        return self
    
def layers2odds(layers):
    odds_layers = []
    for i, layer in enumerate(layers):
        odds = []
        label = f"l{i}_i" if i else "i"
        for weights, bias in zip(*layer):
            odd = ODD()
            odd.build_odd_rec(weights, -bias, label)
            odds.append(odd)
            odd.clear_cache()
        odds_layers.append(odds)
    return odds_layers

def combine_odds(odds):
    prev_layer = [(odd.bdd, odd.eda_vars) for odd in odds[0]]
    p_vars = prev_layer[0][1]
    for odds_next in odds[1:]:
        next_layer = [(odd.bdd, odd.eda_vars) for odd in odds_next]
        
        switch_layer = []
        for n_bdd, n_vars in next_layer:
            res_bdd = n_bdd.compose({n_var: p_bdd for (p_bdd, _), n_var in zip(prev_layer, n_vars)}) 
            switch_layer.append((res_bdd, p_vars))

        prev_layer = switch_layer
    
    return prev_layer[0]

def compile_nn(net, verbose=False):
    params = list(net.parameters())
    if verbose:
        print("converting to ODDs : ", end="")
        start_convert = time.perf_counter()
        
    odds = layers2odds(zip(params[::2], params[1::2]))
    if verbose:
        print(f"DONE ({time.perf_counter()-start_convert:1.2e})\ncombining ODDs : ", end="")
        start_combine = time.perf_counter()

    res = combine_odds(odds)
    if verbose:
        print(f"DONE ({time.perf_counter()-start_combine:1.2e})")
    return res
    