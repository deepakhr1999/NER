from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
import torch
import string
try:
    from utils import packCharsWithMask, getWordsFrom
except:
    from data.utils import packCharsWithMask, getWordsFrom 


class NERDataset(Dataset):
    def __init__(self, sourceName, targetName):
        """Reads the source file and returns sentences and sorted unique words
        Args:
            filename(str) : name of the source file of the dataset
        """
        with open(sourceName, 'r') as sourceFile, open(targetName, 'r') as targetFile:
            pairs = [
                (sourceLine.strip(), targetLine.strip())
                for sourceLine, targetLine in zip(sourceFile, targetFile)
                if sourceLine.strip() != '-DOCSTART-'
            ]

        sentences, targets = zip(*pairs)

        # make char_dict and convert sentence into char double arrays
        self.charIdx = {c:i for i, c in enumerate(string.printable)}
        self.chars = [
            [
                [self.charIdx[c] for c in word]
                for word in line.split()
            ]
            for line in sentences
        ]

        # extract unique words and tags from the document
        self.words = getWordsFrom(sentences)
        self.tags  = getWordsFrom(targets)
        
        # self.pack_collate function is implemented using numTags for one hot encoding
        self.numTags = len(self.tags)

        # make word_dict and convert sentence to list of indices
        self.wordIdx = {word: i+2 for i, word in enumerate(self.words)}
        self.wordIdx['PAD'] = 0
        self.wordIdx['UNK'] = 1
        self.sentences = [
            torch.LongTensor([self.wordIdx[word] for word in sentence.split()])
            for sentence in sentences
        ]

        # make target_dict and convert targets to list of indices
        self.tagIdx = {tag: i for i, tag in enumerate(self.tags)}
        self.targets = [
            torch.LongTensor([self.tagIdx[tag] for tag in line.split()])
            for line in targets
        ]


    def __len__(self):
        return len(self.sentences)

    def getLengths(self):
        return [len(s) for s in self.sentences]
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.chars[idx], self.targets[idx]

    def pack_collate(self, batch):
        (words, chars, targets) = zip(*batch)
        lens = [len(x) for x in words]

        words = pad_sequence(words)
        words = pack_padded_sequence(words, lens, enforce_sorted=False)

        targets = pad_sequence(targets, batch_first=False)
        targets = pack_padded_sequence(targets, lens, enforce_sorted=False)

        chars, mask = packCharsWithMask(chars)
        return words, chars, mask, targets

    def getLoader(self, tokenCap=4096):
        sampler = TokenSampler(self.getLengths(), tokenCap)
        return DataLoader(self, batch_sampler=sampler, collate_fn=self.pack_collate)

class TokenSampler(Sampler):
    def __init__(self, lengths, tokenCap):
        self.lengths = lengths
        self.tokenCap = max(tokenCap, max(lengths))
    
    def __len__(self):
        return sum(self.lengths) // self.tokenCap + 1
    
    def __iter__(self):
        """
            Constructs batches with indices such that
            sum(lengths[idx] for idx in batch) is as large as possible but less than tokenCap
        """
        
        # buffer stores random permutation of indices
        self.buffer = torch.randperm(len(self.lengths))
        
        batch = []
        runningSum = 0
        for idx in self.buffer:
            if runningSum + self.lengths[idx] > self.tokenCap:
                yield batch
                batch = []
                runningSum = 0
            
            batch.append(idx)
            runningSum += self.lengths[idx]
        
        if len(batch) != 0:
            yield batch

if __name__ == '__main__':
    sourceName = 'data/conll03/eng.train.src'
    targetName = 'data/conll03/eng.train.trg'

    data = NERDataset(sourceName, targetName)
    loader = data.getLoader(150)
    
    words, chars, mask, targets = next(iter(loader))

    print("----Shapes----")
    print(words.data, chars.data, targets.data, sep='\n')

    print("----Batching----")
    loader = data.getLoader(4096)

    g = []
    for i,batch in enumerate(loader):
        words, chars, mask, targets = batch
        x = words.data.numel()
        g.append(x)
        
    print(sum(g), sum(data.getLengths()))