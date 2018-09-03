import time
import re

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

def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
    """ Put value as 0th element in OrderedDict

    By Ashwini Chaudhary
    https://stackoverflow.com/a/16664932/3613853

    """
    if hasattr(dct, 'move_to_end'):
        dct[key] = value
        dct.move_to_end(key, last=False)
    else: # Before Python3.2
        root = dct._OrderedDict__root
        first = root[1]

        if key in dct:
            link = dct._OrderedDict__map[key]
            link_prev, link_next, _ = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            link[0] = root
            link[1] = first
            root[1] = first[0] = link
        else:
            root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
            dict_setitem(dct, key, value)

def parse_dataset_name(store_name):
    unstab, set, gen, dim, dataset, filter = re.split('(?:(unstable)_|)(sane|test|training)_(?:gen(\d+)_|)(\d+)D_(.*)_filter(\d+).h5', store_name)[1:-1]
    if filter_id is not None:
        filter_id = int(filter_id)
    gen = int(gen)
    dim = int(dim)
    if unstab == 'unstable':
        unstable = True
    elif unstab == '':
        unstable = True

    return unstable, set, gen, dim, dataset, filter_id
