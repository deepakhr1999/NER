import itertools
import string
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

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

def getWordsFrom(sentences):
    # split each sentence
    separated = map(str.split, sentences)
    concatenated = itertools.chain(*separated)
    words = list(set(concatenated))
    words.sort()
    return words