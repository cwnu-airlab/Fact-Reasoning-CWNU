from seqeval.metrics import classification_report

if __name__=='__main__':
    import os
    import json
    import sys
    import re

    
    filename = sys.argv[1]
    print(filename)
    ext = os.path.splitext(filename)[-1]
    task = sys.argv[2]
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

    if 'token-em' in task:
        with open('data/detail-ner/label_list.txt','r') as f:
            label_dict = [d.strip() for d in f]
            label_dict = {d:i for i,d in enumerate(label_dict)}
        new_pred = list()
        for i in range(len(pred)):
            sub = list()
            for p in pred[i]:
                lbl = ['0']*len(label_dict)
                for l in p:
                    lbl[label_dict[l]] = '1'
                if '1' in lbl: lbl = 'I-'+''.join(lbl)
                else: lbl = 'O'
                sub.append(lbl)
            new_pred.append(sub)
        new_gold = list()
        for i in range(len(gold)):
            sub = list()
            for p in gold[i]:
                lbl = ['0']*len(label_dict)
                for l in p:
                    lbl[label_dict[l]] = '1'
                if '1' in lbl: lbl = 'I-'+''.join(lbl)
                else: lbl = 'O'
                sub.append(lbl)
            new_gold.append(sub)
        gold = new_gold.copy()
        pred = new_pred.copy()

    elif 'token' in task:
        new_gold = list()
        new_pred = list()
        for i in range(len(gold)):
            lbl = sorted(set([y for x in gold[i]+pred[i] for y in x]))
            for l in lbl:
                ner_tag = re.sub('^I_','I-',l.replace('-','_'))
                temp_gold = [ner_tag if l in d else 'O' for d in gold[i]]
                temp_pred = [ner_tag if l in d else 'O' for d in pred[i]]
                new_gold.append(temp_gold.copy())
                new_pred.append(temp_pred.copy())

        gold = new_gold.copy()
        pred = new_pred.copy()

#         labels = sorted(set(gold + pred))
#         with open('../data/io-tag_label_list.txt','r') as f:
#             labels = [d.strip() for d in f]
#         labels.remove('O')
    print('sample size:', len(gold),'/',len(pred))
    print(classification_report(gold,pred, digits=4))

