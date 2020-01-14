import os
import pickle

def from_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None

def to_cache(cache_file, data):
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def memoize(func, cache_file, refresh=False):
    def memoized_func(*args):
        result = from_cache(cache_file)
        if result is None or refresh is True:
            result = func(*args)
            to_cache(cache_file, result)
        return result
    return memoized_func

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
