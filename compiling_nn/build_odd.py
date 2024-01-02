import sys
import time
import random
import signal
from graphviz import Digraph

infty = float('inf')

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
            else:
                print(f"Done in {deltatime}ms")
            return res
        return wrapper
    return inner

# CLASS DEFINITION

class Node:
    def __init__(self, interval=None, name=None):
        self.child = dict()
        self.parent = dict()
        self.interval = interval
        self.name = name
        self.id = f"n{id(self)}"

    def __str__(self):
        return f"{self.child.__len__()} children | {self.parent.__len__()} parents | {self.interval:.2f}"
    
    def add_child(self, other, label):
        self.child.update({label: other})
        other.parent.update({label: self})

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

# RECURSIVE VERSION

@timecounter(message="Built recursively in {:.2f}ms")
@timelimit()
def build_odd_rec(weights_list, threshold):
    n = len(weights_list)
    one_sink = Node(Interval(threshold, infty, closed_right=False))
    store_in_cache(n, one_sink)
    zero_sink = Node(Interval(-infty, threshold, closed_left=False, closed_right=False))
    store_in_cache(n, zero_sink)
    return build_sub_odd_rec(weights_list, 0, 0)

def build_sub_odd_rec(weights, k, v):
    node = Node(Interval(-infty, infty, closed_left=False, closed_right=False))
    # print(f"enter\t  k: {k} | v:{v: d} | {node}")
    weight = weights[k]
    for e in {0, 1}:
        w = e*weight
        v_child = v + w
        child = find_in_cache(k+1, v_child)
        if child is None:
            child = build_sub_odd_rec(weights, k+1, v_child)
        node.add_child(child, e)
        node.interval.intersect(child.interval-w)
        # print(f"intersect k: {k} | v:{v: d} | {node} | {v_child} | {child.interval:.2f}")
    store_in_cache(k, node)
    # print(f"exit\t  k: {k} | v:{v: d} | {node}")
    return node

# ITERATIVE VERSION
# Need to be fixed / maybe not useful

@timecounter(message="Built iteratively in {:.2f}ms")
@timelimit()
def build_odd_iter(weights_list, threshold):
    n = len(weights_list)
    one_sink = Node(Interval(threshold, infty, closed_right=False))
    store_in_cache(n, one_sink)
    zero_sink = Node(Interval(-infty, threshold, closed_left=False, closed_right=False))
    store_in_cache(n, zero_sink)

    stack = []
    root = Node(Interval(-infty, infty, closed_left=False, closed_right=False))
    stack.append((root, 0, 0, 0))
    stack.append((root, 1, 0, 0))

    while stack:
        node, e, k, v = stack.pop()

        w = e * weights_list[k]
        v_child = v + w
        child = find_in_cache(k+1, v_child)

        if child is None:
            child = Node(Interval(-infty, infty, closed_left=False, closed_right=False))
            stack.append((child, 0, k+1, v_child))
            stack.append((child, 1, k+1, v_child))

        node.add_child(child, e)
        #FIXME child.interval is not correct because it needs to be intersected by the next iteration
        #2eme stack en parallèle qui contient ce qu'il faut update 
        node.interval.intersect(child.interval - w)
        store_in_cache(k+1, node)

    return root

# CACHE RELATED

cache = dict()

def reset_cache():
    cache.clear()

def store_in_cache(k, node):
    if cache.get(k):
        cache[k].append(node)
    else:
        cache.update({k: [node]})

def find_in_cache(k, value):
    cache_line = cache.get(k)
    if cache_line:
        for node in cache_line:
            if value in node.interval:
                return node
    return None

def cache_total_node_count():
    node_count = 0
    for x in cache.values():
        node_count+=x.__len__()
    print("cache node count :", node_count)

# DRAW GRAPH

def make_graph_from_cache(name=None):
    dot = Digraph()
    for cache_line in cache.values():
        for node in cache_line:
            dot.node(node.id, f"{node.interval:.2f}")
            for e, child in node.child.items():
                dot.edge(node.id, child.id, str(e))

    name = name if name else "odd"
    dot.render(filename=f"odd/{name}")

# MAIN

def main():
    try:
        n_weights = int(sys.argv[1])
    except Exception:
        n_weights = 10
    weights = [(random.random()-.5)*5 for _ in range(n_weights)] # 27 in 10s / 30 ~ 1min
    # weights = [1, -1, 2]
    threshold = 0

    odd = build_odd_rec(weights, threshold)
    print(odd)
    cache_total_node_count()
    if n_weights < 11:
        make_graph_from_cache("random")

    # reset_cache()
    # odd = build_odd_iter(weights, threshold)
    # print(odd)
    # cache_total_node_count()

    reset_cache()
    weights = list(sorted(weights))
    odd = build_odd_rec(weights, threshold)
    print(odd)
    cache_total_node_count()
    if n_weights < 11:
        make_graph_from_cache("ascending")

    reset_cache()
    weights = weights[::-1]
    odd = build_odd_rec(weights, threshold)
    print(odd)
    cache_total_node_count()
    if n_weights < 11:
        make_graph_from_cache("descending")

if __name__ == "__main__":
    main()
