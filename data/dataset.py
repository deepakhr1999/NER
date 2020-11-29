from torch.utils.data import Dataset, DataLoader
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

    def __getitem__(self, idx):
        return self.sentences[idx], self.chars[idx], self.targets[idx]

    def pack_collate(self, batch):
        (words, chars, targets) = zip(*batch)
        lens = [len(x) for x in words]

        words = pad_sequence(words)
        words = pack_padded_sequence(words, lens, enforce_sorted=False)

        targets = pad_sequence(targets, batch_first=False)
        targets = pack_padded_sequence(targets, lens, enforce_sorted=False)
        b = targets.data.size(0)
        oneHot = torch.zeros(b, self.numTags)
        oneHot[torch.arange(b), targets.data] =  1.
        _, *args = targets
        targets = PackedSequence(oneHot, *args)

        chars, mask = packCharsWithMask(chars)
        return words, chars, mask, targets

    def getLoader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.pack_collate)


if __name__ == '__main__':
    sourceName = 'data/conll03/eng.train.src'
    targetName = 'data/conll03/eng.train.trg'

    loader = NERDataset(sourceName, targetName).getLoader(batch_size=3)
    
    words, chars, mask, targets = next(iter(loader))

    print(words.data, chars.data, targets.data, sep='\n')
        
# class NERdata(Dataset):
#   def __init__(self,src_path,trg_path):
#     self.trg_path = trg_path
#     self.src_path = src_path
#     self.src_data=list()
#     self.sentences = list()
#     self.word_dict = dict()
#     self.words = list()
#     self.unique_words = set()
#     self.trg_data = list()
#     self.labels = list()
    
#     self._init()

#   def __len__(self):
#     return len(self.unique_words)
  
#   def __getitem__(self,idx):
#     return self.word_dict[idx]
  
#   def _init(self):

#     src_inp = open(self.src_path)
#     trg_inp = open(self.trg_path)
#     for i in src_inp:
#       i = i.strip()
#       self.src_data.append(i)
#       if i!= '-DOCSTART-':
#         self.sentences.append(i)
#         for word in i.split(' '):
#           self.unique_words.add(word)
#     src_inp.close()
#     for i in self.unique_words:
#       self.words.append(i)
#     self.words.sort()

#     self.word_dict = {word:(i+2) for i,word in enumerate(self.words)}
#     self.word_dict['PAD'] = 0
#     self.word_dict['UNK'] = 1

#     label_set = set()
#     sent_labels = []
#     for i,j in enumerate(trg_inp):
#       l = []
#       j = j.strip()
#       self.trg_data.append(j)
#       if self.src_data[i] != '-DOCSTART-':
#         for k in j.split(' '):  
#           l.append(k)
#           label_set.add(k)
#         sent_labels.append(l)
#     trg_inp.close()

#     all_labels = []
#     for i in label_set:
#       all_labels.append(i)
#     all_labels.sort()

#     print(all_labels)
#     # label_len = len(all_labels)
    
#     for word_lab in sent_labels:
#       l1 = []
#       for i in word_lab:
#         temp = np.zeros(len(all_labels))
#         temp[all_labels.index(i)] = 1
#         l1.append(temp)

#       self.labels.append(l1)    

