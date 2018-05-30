import time
def str_to_int_or_float(string):
    if not isinstance(string):
        raise ValueError('Please input a string')
    try:
        result = int(string)
    except ValueError:
        result = float(string)
    return result

def first(s):
    '''Return the first element from an ordered collection
       or an arbitrary element from an unordered collection.
       Raise StopIteration if the collection is empty.
    '''
    return next(iter(s.items()))

def profile(x):
    """ Placeholder decorator for memory profiling """
    return x

def notify_task_done(task, starttime=None):
    msg = '{!s} done'.format(task)
    if starttime != None:
        msg += ' after {:.0f}s'.format(time.time() - starttime)
    print(msg)
