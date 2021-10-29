# coding=utf-8
import pickle
import os


def save_cache(cache_path, obj):
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)


def load_cache(cache_path, create_func=None):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        if create_func is None:
            raise Exception("cache does not exist: ", cache_path)
        obj = create_func()
        save_cache(cache_path, obj)
        return obj
