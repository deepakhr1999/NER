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
import conlleval_perl as conll

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='pytorchlightning ckpt path', required=True)
    parser.add_argument('--type', help='testa or testb', default='testa')
    parser.add_argument('--beam', help='beamsize for decoding', default=4, type=int)
    parser.add_argument('--file', default='results/temp.txt', help="conlleval input")
    parser.add_argument('--cuda', default=False, action='store_true', help='use gpu')
    parser.add_argument(
        "-l", "--latex",
        default=False, action="store_true",
        help="generate LaTeX output"
    )
    parser.add_argument(
        "-r", "--raw",
        default=False, action="store_true",
        help="accept raw result tags"
    )
    parser.add_argument(
        "-d", "--delimiter",
        default=None,
        help="alternative delimiter tag (default: single space)"
    )
    parser.add_argument(
        "-o", "--oTag",
        default="O",
        help="alternative delimiter tag (default: O)"
    )
    return parser



def main(args):
    # load dataset
    sourceName = 'data/conll03/eng.train.src'
    targetName = 'data/conll03/eng.train.trg'
    gloveFile = 'data/conll03/trimmed.300d.Cased.txt'
    symbFile = 'data/conll03/sym.glove'
    testSrc = f'data/conll03/eng.{args.type}.src'
    testTrg = f'data/conll03/eng.{args.type}.trg'
    predFile = f"preds/{args.type}_beam{args.beam}_{os.path.basename(args.ckpt)}"
    beamSize = args.beam
    data = NERDataset(sourceName, targetName, gloveFile, symbFile)
    data.readTestFile(testSrc, testTrg)
    loader = data.getLoader(4096, shuffle=False)


    # load model
    prevCheckpointPath = args.ckpt
    with open('config.json', 'r') as file:
        kwargs = json.load(file)
        
    model = GlobalContextualDeepTransition.load_from_checkpoint(prevCheckpointPath, **kwargs)
    model = model.eval()
    if args.cuda:
        model = model.cuda()


    # perform beamsearch
    tester = BeamSearcher(beamSize=beamSize, model=model)

    out = []
    for batch in tqdm(loader, desc="Performing BeamSearch"):
        if args.cuda:
            batch = (w.cuda() for w in batch)
        preds = tester(batch)
        out.extend(preds)

    tester.writePreds(out, data.tags, predFile, testSrc)
    tester.getResultFile(testSrc, predFile, testTrg, args.file)
    acc, prec, rec, f1 = conll.main(args)
    return acc, prec, rec, f1

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    acc, prec, rec, f1 = main(args)
    print("Accuracy: ", acc)
    print("precision:", prec)
    print("recall:   ", rec)
    print("f1-score: ", f1)