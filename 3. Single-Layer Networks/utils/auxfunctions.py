

def frange(start, stop, step):
    i = start
    _range = []
    while i < stop:
        _range.append(i)
        i += step
    return _range