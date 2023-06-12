import os
import json
import sys
import sklearn.metrics as met

if __name__=='__main__':
    
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[-1]
    if ext == '.txt':
        with open(filename, 'r') as f:
            data = [d.strip() for d in f.readlines()]
        columns = data[0]
        
        pred = [d.replace('PRED:','').strip() for d in data if 'PRED:' in d]
        gold = [d.replace('GOLD:','').strip() for d in data if 'GOLD:' in d]
        srce = [d.replace('SRCE:','').strip() for d in data if 'SRCE:' in d]
    elif ext == '.jsonl':
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]
        pred = [d['output']['pred'] for d in data if 'output' in d]
        gold = [d['output']['gold'] for d in data if 'output' in d]
        srce = [d['output']['srce'] for d in data if 'output' in d]

    assert len(pred) == len(gold) and len(gold) == len(srce)
    print('data size: {}\n'.format(len(gold)))

    print(met.classification_report(gold, pred, digits=4))
