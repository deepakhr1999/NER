import scripts.test as tester

parser = tester.getParser()
args = parser.parse_args(['--ckpt', 'lightning_logs/rasputin.ckpt', '--beam', '8'])
loader, tags = tester.getLoader(args)
acc, prec, rec, f1 = tester.main(args, loader, tags)
print("Accuracy: ", acc)
print("precision:", prec)
print("recall:   ", rec)
print("f1-score: ", f1)