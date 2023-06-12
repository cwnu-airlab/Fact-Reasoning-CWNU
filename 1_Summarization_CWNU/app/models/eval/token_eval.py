
def get_cls(pred, gold, labels=None):
    import sklearn.metrics as met

    print(met.classification_report(gold, pred, digits=4, labels=labels))
    exit()
    print()

    targets = sorted(list(set(gold)))
    macro_precision = met.precision_score(gold, pred, average = 'macro', zero_division=0)
    macro_recall = met.recall_score(gold, pred, average = 'macro', zero_division=0)
    macro_f1 = met.f1_score(gold, pred, average = 'macro', zero_division=0)
    micro_f1 = met.f1_score(gold, pred, average = 'micro', zero_division=0)
    weighted_f1 = met.f1_score(gold, pred, average = 'weighted', zero_division=0)
    accuracy = met.accuracy_score(gold, pred)

    precisions = met.precision_score(gold, pred, average = None, labels = targets, zero_division=0)
    recalls = met.recall_score(gold, pred, average = None, labels = targets, zero_division=0)
    f_measures = met.f1_score(gold, pred, average = None, labels = targets, zero_division=0)

    print("class\tprecision\trecall\tf1-score")
    for i, target in enumerate(targets):
        print("%s\t%0.3f\t%0.3f\t%0.3f"%(target,precisions[i],recalls[i],f_measures[i]))
    print()
    print("%s\t%0.3f\t%0.3f\t%0.3f"%("MACRO",macro_precision,macro_recall,macro_f1))
    print("Accuracy\t%0.3f"%accuracy)
    print()
    print("%s\t%0.3f\t%0.3f\t%0.3f"%("MEAN",sum(precisions)/len(precisions),sum(recalls)/len(recalls),sum(f_measures)/len(f_measures)))
    print()

    result = {'accuracy':accuracy, 'macr-f1':macro_f1, 'micro-f1':micro_f1, 'weighted_f1':weighted_f1}
    return result


if __name__=='__main__':
    import os
    import json
    import sys
    
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

    if 'class' in task:
        get_cls(gold, pred)
    elif 'token-em' in task:
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
                lbl = 'I-'+''.join(lbl)
                sub.append(lbl)
            new_pred.append(sub)
        new_gold = list()
        for i in range(len(gold)):
            sub = list()
            for p in gold[i]:
                lbl = ['0']*len(label_dict)
                for l in p:
                    lbl[label_dict[l]] = '1'
                lbl = 'I-'+''.join(lbl)
                sub.append(lbl)
            new_gold.append(sub)
        gold = new_gold.copy()
        pred = new_pred.copy()
        gold = [y for x in gold for y in x]
        pred = [y for x in pred for y in x]
        print('tag size:', len(gold))
        outside = 'I-'+''.join(['0']*len(label_dict))
        labels = sorted(set(gold+pred))
        labels.remove(outside)
        get_cls(gold, pred, labels=labels)
    elif 'token' in task:
        gold = [y for x in gold for y in x]
        pred = [y for x in pred for y in x]

        for i in range(len(gold)):
            if gold[i] != [] or pred[i] != []:
                temp = [[d,d] if d in gold[i] else [d,'O'] for d in pred[i]]
                temp += [['O',d] for d in gold[i] if d not in pred[i]]
                pred[i] = [d[0] for d in temp]
                gold[i] = [d[1] for d in temp]
            elif gold[i] == [] and pred[i] == []:
                gold[i] = ['O']
                pred[i] = ['O']

        gold = [y for x in gold for y in x]
        pred = [y for x in pred for y in x]
        print('tag size:', len(gold),'/',len(pred))

        labels = sorted(set(gold + pred))
        labels.remove('O')
        get_cls(pred, gold, labels=labels)

