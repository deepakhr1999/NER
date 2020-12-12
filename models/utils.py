import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence
from torch.nn import init
import math

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

def getSignal(t, u, timeBias, device):
    n = u//2
    logTimeScaleInc = math.log(1e4)/ (n - 1)
    positions = torch.arange(t, dtype=torch.float, device=device) + timeBias
    invTimeScale = torch.exp(torch.arange(n, dtype=torch.float, device=device) * -logTimeScaleInc)
    scaledTime = torch.unsqueeze(positions, 1) * torch.unsqueeze(invTimeScale, 0) # shape (t, u/2)
    signal = torch.cat([torch.sin(scaledTime), torch.cos(scaledTime)], axis=1) # shape (t, u)
    return signal

def addTimeSignal(sequence:PackedSequence, device, timeBias:float=.0)->PackedSequence:
    """
        Takes in a PackedSequence, adds the time signal and returns it
    """
    padded, lens = pad_packed_sequence(sequence)
    t, b, u = padded.shape
    signal = getSignal(t, u, timeBias, device) # t u
    signal = torch.unsqueeze(signal, 1) # shape (t, 1, u)
    padded = padded + signal
    result = pack_padded_sequence(padded, lens, enforce_sorted=False)
    return result

def packCharsWithMask(sequences):
    """Takes in a 3d list with axes: sentence, word, char
    
        Returns result, mask
        result is a 3d tensor of shape sentences, max_sentence_length, max_word_length as a packed sequence
        mask has -1e9 where there is a zero padded character in a word
    """
    b = len(sequences)
    w = max(len(word) for sentence in sequences for word in sentence)
    
    lengths = [len(sentence) for sentence in sequences]
    t = max(lengths)
    
    result = torch.zeros(t, b, w, dtype=torch.long)
    for i, sentence in enumerate(sequences):
        for j, word in enumerate(sentence):
            result[j, i, :len(word)] = torch.LongTensor(word)

    result = pack_padded_sequence(result, lengths, enforce_sorted=False)
    
    mask = (result.data == 0) * -1e9
    mask = torch.unsqueeze(mask, -1)
    
    return result, mask

def recursiveXavier(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # xavier init for conv and linear layers
        init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        # For layer norm, we need to init scale = 1 and offset = 0
        # use only normal
        init.normal_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
