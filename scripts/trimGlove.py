"""
    Used for trimming glove cased file to smaller file based on conll dataset vocab
    Usage: python scripts/trimGlove.py <srcfile> <glovefile> <trimmedoutfile>
    Eg: python .\scripts\trimGlove.py .\data\conll03\eng.train.src E:\glove\glove.840B.300d.txt .\data\conll03\trimmed.300d.Cased.txt
"""
import sys
from tqdm import tqdm

def getVocab(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '-DOCSTART-':
                continue
            for word in line.split():
                words.append(word)
    return set(words)            
            
def trimGlove(gloveFile, outFile, vocab):
    visited = set()
    with open(gloveFile, 'r', encoding='utf-8') as file, open(outFile, 'w') as fout:
        expectedLength = 2196017
        for line in tqdm(file, desc='Reading glove file', total=expectedLength):
            line = line.strip()
            idx = line.find(' ')
            word = line[:idx]
            if word not in visited and word in vocab:
                fout.write(line + '\n')
                visited.add(word)

if __name__ == '__main__':
    dataFile, gloveFile, trimmedFile = sys.argv[1:]
    vocab = getVocab(dataFile)
    trimGlove(gloveFile, trimmedFile, vocab)