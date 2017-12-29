from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt='Elapsed'):
    print fmt + ' - %s s' % (time() - _tstart_stack.pop())