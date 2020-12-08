import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS

from data.dataset import NERDataset
from models.networks import GlobalContextualDeepTransition
from torch.nn.utils.rnn import pad_packed_sequence
import re

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default

sourceName = 'data/conll03/eng.train.src'
targetName = 'data/conll03/eng.train.trg'
gloveFile = 'data/conll03/trimmed.300d.Cased.txt'
symbFile = 'data/conll03/sym.glove'
data = NERDataset(sourceName, targetName, gloveFile, symbFile)

numChars = 100
charEmbedding = 128
numWords = len(data.wordIdx)
wordEmbedding = 300
contextOutputUnits = 128
contextTransitionNumber = transitionNumber = 4
encoderUnits = 256
decoderUnits = 256
prevCheckpointPath = 'lightning_logs/version_0/checkpoints/epoch=100.ckpt'
kwargs = dict(numChars=numChars, charEmbedding=charEmbedding, numWords=numWords,
                 wordEmbedding=wordEmbedding, contextOutputUnits=contextOutputUnits, contextTransitionNumber=contextTransitionNumber,
                    encoderUnits=encoderUnits, decoderUnits=decoderUnits, transitionNumber=transitionNumber, numTags=data.numTags)
model = GlobalContextualDeepTransition.load_from_checkpoint(prevCheckpointPath, **kwargs)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/<sentence>')
def forward(sentence):
    inp = re.sub(r'[^\w\s]', '', sentence)
    inp = data.encodeSentence(inp)
    out = model.testForward(inp)
    out, lens = pad_packed_sequence(out)
    out = out.view(-1).numpy()
    out = ' '.join([data.tags[i] for i in out])
    return Response(out)

if __name__ == '__main__':
    app.run(debug=True)