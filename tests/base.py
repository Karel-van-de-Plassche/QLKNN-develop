from unittest import TestCase
from IPython import embed

def skip_if(expr):
    def decorator(method):
        @wraps(method)
        def inner(self):
            should_skip = expr() if callable(expr) else expr
            if not should_skip:
                return method(self)
            elif VERBOSITY > 1:
                print('Skipping %s test.' % method.__name__)
        return inner
    return decorator


def skip_unless(expr):
    return skip_if((lambda: not expr()) if callable(expr) else not expr)


def skip_case_if(expr):
    def decorator(klass):
        should_skip = expr() if callable(expr) else expr
        if not should_skip:
            return klass
        elif VERBOSITY > 1:
            print('Skipping %s test.' % klass.__name__)
            class Dummy(object): pass
            return Dummy
    return decorator


def skip_case_unless(expr):
    return skip_case_if((lambda: not expr()) if callable(expr) else not expr)
