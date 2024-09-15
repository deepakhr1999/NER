"""
python train.py --base_dir data/conll03 \
    --config_file configs/config.json \
    --prev_checkpoint lightning_logs/version_7/epoch=502-step=24938.ckpt \
    --epochs 1000
"""

import argparse
import json
import os
import pytorch_lightning as pl
from data.dataset import NERDataset
from models.networks import GlobalContextualDeepTransition
from models.utils import SaveEachEpoch


def parse_args():
    "args parsed here"
    parser = argparse.ArgumentParser(
        description="Train a Global Contextual Deep Transition model."
    )

    # File paths
    parser.add_argument(
        "--base_dir", type=str, default="data/conll03", help="Base directory for data."
    )
    parser.add_argument(
        "--source_file", type=str, default="eng.train.src", help="Source training file."
    )
    parser.add_argument(
        "--target_file", type=str, default="eng.train.trg", help="Target training file."
    )
    parser.add_argument(
        "--glove_file",
        type=str,
        default="trimmed.300d.Cased.txt",
        help="GloVe embeddings file.",
    )
    parser.add_argument(
        "--symb_file",
        type=str,
        default="sym.glove",
        help="Symbol file for GloVe embeddings.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/config.json",
        help="Model configuration file.",
    )

    # Checkpoint and logging
    parser.add_argument(
        "--prev_checkpoint",
        type=str,
        default=None,
        help="Path to previous checkpoint to resume training.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs/backup/",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train."
    )

    return parser.parse_args()


def main():
    "Driver code"
    args = parse_args()

    # Paths
    base = args.base_dir
    source_name = os.path.join(base, args.source_file)
    target_name = os.path.join(base, args.target_file)
    glove_file = os.path.join(base, args.glove_file)
    symb_file = os.path.join(base, args.symb_file)

    # Load dataset
    data = NERDataset(source_name, target_name, glove_file, symb_file)
    loader = data.getLoader(4096)

    # Load model config
    with open(args.config_file, "r", encoding="utf-8") as file:
        kwargs = json.load(file)
    print("Init model params =", json.dumps(kwargs, indent=4))

    # Initialize model and weights
    model = GlobalContextualDeepTransition(**kwargs)
    model.init_weights(data.embeddingWeights)

    # Save checkpoints every epoch
    ckpt = SaveEachEpoch(dirpath=args.log_dir, filename="ckpt{epoch:02d}", period=1)

    # Initialize trainer
    trainer = pl.Trainer(
        resume_from_checkpoint=args.prev_checkpoint,
        callbacks=[ckpt],
        gradient_clip_val=5.0,
        gpus=1,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=10,
    )

    # Train the model
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
