import json
import sys
import os
sys.path.append(os.path.join(sys.path[0], ".."))
from data.dataset import NERDataset
from models.networks import GlobalContextualDeepTransition
from models.utils import SaveEachEpoch
import pytorch_lightning as pl


base = 'data/conll03'
sourceName = f'{base}/eng.train.src'
targetName = f'{base}/eng.train.trg'
gloveFile = f'{base}/trimmed.300d.Cased.txt'
symbFile = f'{base}/sym.glove'
prevCheckpoint = None#'lightning_logs/version_7/epoch=502-step=24938.ckpt'
data = NERDataset(sourceName, targetName, gloveFile, symbFile)
loader = data.getLoader(4096)


with open('config.json', 'r') as file:
    kwargs = json.load(file)
print("Init model params =", json.dumps(kwargs, indent=4))
model = GlobalContextualDeepTransition(**kwargs)
model.init_weights(data.embeddingWeights)

numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {numParams:,}") # 7,313,34


ckpt = SaveEachEpoch(
    dirpath='lightning_logs/backup/',
    filename='ckpt{epoch:02d}',
    period=1
)

trainer = pl.Trainer(resume_from_checkpoint=prevCheckpoint, callbacks=[ckpt],
                        gradient_clip_val=5., gpus=1, max_epochs=500, progress_bar_refresh_rate=10)
trainer.fit(model, loader)





