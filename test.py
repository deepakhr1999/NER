"""
    command to run in windows: Get-Content result.txt | python conlleval_perl.py
"""

import json
import torch
import pickle
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from data.dataset import NERDataset
from models.networks import GlobalContextualDeepTransition
from scripts.beamsearch import BeamSearcher

parser = argparse.ArgumentParser()
parser.add_argument('--type', help='testa or testb', required=True)
parser.add_argument('--ckpt', help='pytorchlightning ckpt path', required=True)
parser.add_argument('--beam', help='beamsize for decoding', default=4, type=int)
args = parser.parse_args()

print(args)

sourceName = 'data/conll03/eng.train.src'
targetName = 'data/conll03/eng.train.trg'
gloveFile = 'data/conll03/trimmed.300d.Cased.txt'
symbFile = 'data/conll03/sym.glove'
testSrc = f'data/conll03/eng.{args.type}.src'
testTrg = f'data/conll03/eng.{args.type}.trg'
predFile = 'preds/' + os.path.basename(testTrg) + '.out'
beamSize = args.beam
data = NERDataset(sourceName, targetName, gloveFile, symbFile)
data.readTestFile(testSrc, testTrg)
loader = data.getLoader(4096, shuffle=False)

prevCheckpointPath = args.ckpt

with open('config.json', 'r') as file:
    kwargs = json.load(file)
    
model = GlobalContextualDeepTransition.load_from_checkpoint(prevCheckpointPath, **kwargs)
model = model.eval().cuda()

tester = BeamSearcher(beamSize=beamSize, model=model)

out = []
for batch in tqdm(loader, desc="Performing BeamSearch"):
    batch = (w.cuda() for w in batch)
    preds = tester(batch)
    out.extend(preds)

tester.writePreds(out, data.tags, predFile, testSrc)

result = 'results/'+ os.path.basename(prevCheckpointPath) + '.' + f'beamsize={beamSize}.' + os.path.basename(predFile)
print('Saving file at ', result)
tester.getResultFile(testSrc, predFile, testTrg, result)