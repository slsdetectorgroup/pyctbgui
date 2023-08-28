from functools import wraps
from time import time


def timing(print_args=False):

    def _timing(f):

        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            if print_args:
                print(f'func:{f.__name__} args:[{args}, {kw}] took: {te - ts} sec')
            else:
                print(f'func:{f.__name__}  took: {te - ts} sec')

            return result

        return wrap

    return _timing
