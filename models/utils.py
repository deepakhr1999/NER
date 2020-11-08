import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class Namespace:
    """
        reference: https://www.tutorialspoint.com/How-do-I-create-a-Python-namespace
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def reverse_packed_sequence(backwardOutput):
    """
    Returns the reversed PackedSequence
    """
    # unpack (note that this returns a new tensor so any change in operations here dont affect input)
    backwardOutput, lengths = pad_packed_sequence(backwardOutput)
    
    # reverse ie, [1 2 3 0 0] - > [3 2 1 0 0]
    # expects input in the form of t b u
    b = backwardOutput.size(1)
    # backwardOutput = backwardOutput.clone()
    for i in range(b):
        # valid_slice is data[:lengths[i], i, :]
        backwardOutput[:lengths[i], i] = torch.flip(backwardOutput[:lengths[i], i], [0])

    # pack back
    backwardOutput = pack_padded_sequence(backwardOutput, lengths, enforce_sorted=False)
    
    return backwardOutput

def param(*args):
    """
    Returns a torch param init with @args
    initialized with U(-.8, +.8)
    """
    m = torch.nn.Parameter( torch.Tensor(*args) )
    m.data.uniform_(-0.8, 0.8)
    return m

def sample_sequence(input_units, lengths):
    x = [torch.rand(l, input_units) for l in lengths]
    y = [torch.flip(s, [0]) for s in x ]

    x = pack_padded_sequence(pad_sequence(x), lengths, enforce_sorted=False)
    y = pack_padded_sequence(pad_sequence(y), lengths, enforce_sorted=False)
    
    return x, y