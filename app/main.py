import os,sys,inspect, json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS

from data.dataset import NERDataset
from models.networks import GlobalContextualDeepTransition
from torch.nn.utils.rnn import pad_packed_sequence
import re
from scripts.beamsearch import BeamSearcher

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default

sourceName = 'data/conll03/eng.train.src'
targetName = 'data/conll03/eng.train.trg'
gloveFile = 'data/conll03/trimmed.300d.Cased.txt'
symbFile = 'data/conll03/sym.glove'
data = NERDataset(sourceName, targetName, gloveFile, symbFile)

with open('config.json', 'r') as file:
    kwargs = json.load(file)
prevCheckpointPath = 'lightning_logs/epoch=475-step=23799.ckpt'

model = GlobalContextualDeepTransition.load_from_checkpoint(prevCheckpointPath, **kwargs)
model.eval()
tester = BeamSearcher(beamSize=4, model=model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/<sentence>')
def forward(sentence):
    processedInput = data.encodeSentence(sentence)
    out = tester(processedInput)[0]
    out = ' '.join([data.tags[i] for i in out])
    return Response(out)

if __name__ == '__main__':
    app.run(debug=True)