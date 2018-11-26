from dependencies import *


def info(data):
    """A quickie helper function that extracts useful information on a numpy array for a quick glance."""
    if str(type(data))[0:13] == "<class 'numpy":   
        print (data.shape, ' ', type(data).__name__, ' ', data.dtype, ' Max: ', np.max(data), ' ', ' Min: ', np.min(data))
    else: 
        print (type(data))
