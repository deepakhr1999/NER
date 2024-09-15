import os
import pandas as pd
from scripts.ckpt_test import getParser, getLoader, main

def load_or_create_metrics_frame(metric_file, columns):
    if os.path.exists(metric_file):
        print("File exists, reading the file")
        return pd.read_csv(metric_file)
    else:
        print("Creating new frame")
        return pd.DataFrame({col: [] for col in columns})

def evaluate_and_update_metrics(args, loader, tags, metrics_frame, columns):
    acc, prec, rec, f1 = main(args, loader, tags)
    row_data = {k: v for k, v in zip(columns, [os.path.basename(args.ckpt), acc, prec, rec, f1])}
    metrics_frame = metrics_frame._append(row_data, ignore_index=True)
    return metrics_frame

ckpt_base = 'lightning_logs/backup/'
parser = getParser()
args = parser.parse_args(["--ckpt", os.path.join(ckpt_base, os.listdir(ckpt_base)[0])])

loader, tags = getLoader(args)
print("Init loader success!")

columns = 'ckpt acc prec recall f1'.split()
metrics_file = os.path.join(args.root, f'{args.type}_beam{args.beam}_backup.csv')
metrics_frame = load_or_create_metrics_frame(metrics_file, columns)

for ckpt in sorted(os.listdir(ckpt_base), key=lambda x: os.path.getmtime(os.path.join(ckpt_base, x))):
    args.ckpt = os.path.join(ckpt_base, ckpt)
    if ckpt not in list(metrics_frame.ckpt):
        metrics_frame = evaluate_and_update_metrics(args, loader, tags, metrics_frame, columns)
        metrics_frame.to_csv(metrics_file, index=False)