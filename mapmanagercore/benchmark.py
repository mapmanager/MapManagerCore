from time import time
import pandas as pd

_times = pd.DataFrame([], columns=["name", "time"])
_disabled = True


def timer(func):
    """
    Time the execution of a function with respect to @timeAll.
    """
    if _disabled:
        return func
    # This function shows the execution time of
    # the function object passed

    def wrap_func(*args, **kwargs):
        global _times
        t1 = time() * 1000
        result = func(*args, **kwargs)
        t2 = time() * 1000
        name = func.__name__
        try:
            name = func.__module__ + "." + name
        except:
            pass

        _times.loc[len(_times)] = [name, t2-t1]
        return result

    wrap_func.__name__ = func.__name__
    return wrap_func


def timeAll(func):
    """
    Show the execution time of all functions with a @timer decorator that are
    called by this function.
    """
    if _disabled:
        return func

    # This function shows the execution time of
    # the function object passed

    def q90(x):
        return x.quantile(0.9)

    def wrap_func(*args, **kwargs):
        global _times
        t1 = time() * 1000
        result = func(*args, **kwargs)
        t2 = time() * 1000

        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}ms')
        print(_times.groupby("name")
              .agg(["count", "sum", "median", q90, "max"])
              .sort_values(by=("time", "sum"), ascending=False)
              .to_string())
        _times = pd.DataFrame([], columns=["name", "time"])
        return result

    wrap_func.__name__ = func.__name__
    return wrap_func


def enableBenchmark(enable=True):
    global _disabled
    _disabled = not enable
