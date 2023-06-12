import os
import json
import sys
from nltk import eval
from nltk.translate.meteor_score import meteor_score

def get_rouge(gold, pred):
    null_list = [' ','',[''],[],None]

    result = {1:list(), 2:list(), 3:list(), 'l':list()}
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        for key in result:
            try:
                if p in null_list or g in null_list: raise ZeroDivisionError
                if 'l' == key:
                    result[key].append(eval.rouge_l([g],p))
                else:
                    result[key].append(eval.rouge_n([g],p,key))
            except ZeroDivisionError as e:
                result[key].append(0.0)

    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    for key in result:
        print('rouge-{}: {}'.format(key, result[key]*100), flush=True)
    return result

def get_bleu(gold, pred):
    null_list = [' ','',[''],[],None]

    result = {1:list(), 2:list(), 3:list(), 4:list()}
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        for key in result:
            try:
                if p in null_list or g in null_list: raise ZeroDivisionError
                result[key].append(eval.bleu_n([g],[p],key))
            except ZeroDivisionError as e:
                result[key].append(0.0)

    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    result['a'] = sum([result[key] for key in result])/len(result)
    for key in result:
        print('BLEU-{}: {}'.format(key, result[key]*100), flush=True)
    return result

def get_cider(gold, pred):
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            result.append(float(eval.cider([g],[p])))
        except ZeroDivisionError as e:
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'cider':result}
    print('CIDER: {}'.format(result['cider']*100), flush=True)
    return result

def get_meteor(gold, pred):
    #from nltk import eval
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            #result.append(float(eval.meteor([g],p)))
            result.append(float(meteor_score([g],p)))
        except (IndexError, ZeroDivisionError) as e:
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'meteor':result}
    print('METEOR: {}'.format(result['meteor']*100), flush=True)
    return result


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

    get_bleu(gold, pred)
    get_rouge(gold, pred)
    get_cider(gold, pred)
    get_meteor(gold, pred)

