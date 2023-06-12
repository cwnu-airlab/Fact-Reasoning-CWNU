import re
import json
import torch
import transformers
import random
torch.manual_seed(42)
random.seed(42)

class Service:
    task = [{ 'name': "KG-to-Text",
              'description': 'dummy system' }]

    def __init__(self):
        self.model = Model()
        print('SET model')
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.model.run(content)
            print(ret)
            if 'error' in ret.keys(): return json.dumps(ret), 400
            else: return json.dumps(ret), 200
        except Exception as e:
            return json.dumps({'error': "{}".format(e)}), 400

class Model(object):

    def __init__(self):
        
        config_dict = {
                "en;common-sense":"",
#                 "kr;common-sense":"checkpoint/kr_hotpotqa_gpt2_ms256_mt320",
                "kr;common-sense":"checkpoint/kr_hotpotqa_t5_ms32_mt64",
                "kr;law":"checkpoint/kr_law_ket5-small_ms512_mt256",
                }

        archi_dict = dict()
        archi_dict['kr;law'] = {
                "tokenizer": transformers.T5Tokenizer,
                "model": transformers.T5ForConditionalGeneration,
                }
        archi_dict['kr;common-sense'] = {
                "tokenizer": transformers.T5Tokenizer,
                "model": transformers.T5ForConditionalGeneration,
#                 "tokenizer": transformers.AutoTokenizer,
#                 "model": transformers.GPT2LMHeadModel,
                }

        print('LOAD predicate list', flush=True)
        with open('data/predicate_template.json','r') as f:
            self.predicate_dict = {d['predicate']:d['template'] for d in json.load(f)}
    
        print('LOAD model', flush=True)
        self.system_dict = dict()
        for key in archi_dict:
            if key not in config_dict: continue
            if not config_dict[key]: continue
            self.system_dict[key] = dict()
            self.system_dict[key]['model'] = archi_dict[key]['model'].from_pretrained(config_dict[key]+'/model')
            if key == 'kr;common-sense':
#                 self.system_dict[key]['tokenizer'] = archi_dict[key]['tokenizer'].from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5')
                self.system_dict[key]['tokenizer'] = archi_dict[key]['tokenizer'].from_pretrained(config_dict[key]+'/tokenizer')
            else:
                self.system_dict[key]['tokenizer'] = archi_dict[key]['tokenizer'].from_pretrained(config_dict[key]+'/tokenizer')
        if torch.cuda.is_available():
            for key in self.system_dict: self.system_dict[key]['model'] = self.system_dict[key]['model'].to('cuda')

        
    def get_ids(self, sentence, tokenizer=None, max_length=512):
        input_ids = tokenizer.encode(sentence, max_length=max_length, padding='max_length', truncation=True)
        input_ids = torch.tensor([input_ids])
        return input_ids

    def get_sentence(self, key, inputs):
        tokenizer = self.system_dict[key]['tokenizer']
        model = self.system_dict[key]['model']

        if 'law' in key:
            max_length = 512
            target_length = 256 
        else:
            max_length = 256
            target_length = 320 

        inputs = self.get_ids(inputs, tokenizer=tokenizer, max_length=max_length)
        inputs = inputs.to(model.device)

        predict = model.generate(
                inputs, 
                max_length=target_length, 
                num_beams=5,
                repetition_penalty=5.0,
                )
        
        output = list()
        for pred in predict:
            pred = tokenizer.decode(pred)
#             if '.</s>' not in pred: continue
            pred = re.sub('<pad>','',pred)
            pred = re.sub('</s>','',pred)
            output.append(pred)
        return output


    def run(self, content):
        key = f"{content['question']['language']};{content['question']['domain']}" 
        print(f"TASK: {key}", flush=True)
        question = content['question']['text']
        triples = content.get('triples',None)
        text = [' '.join([d[0],d[1].split(':')[-1],d[2]]) for d in triples if 'no_relation' != d[1]]
        
        inputs = list()
        for _ in range(10):
            sent = ' '.join(text)
            inputs.append(sent)

            random.seed(42)
            random.shuffle(text)
        print(f"INPUT:\n{triples}", flush=True)
        print()
        
        if text is None:
            return {'error':'invalid query'}
        else:
            output = list()
            for text in inputs:
                text = f'{text}'
                #text = f'context: {text} question: {question}'
                out = self.get_sentence(key, text)
                print('PREDICT:',out, flush=True)
                if not out: continue
                output += out
                if len(output) > 1: break
            
            o = self.get_sentence(key, text)
            print()
            print('RESULT:\n'+'\n'.join([f"{[d.strip()]}" for d in output]), flush=True)
            return {'output':output}

if __name__=='__main__':
    
    import sys
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    api_service = Service()
    predict = api_service.do(data)
#     print(json.loads(predict[0]))
#     model = Model()
#     predict = model.run(data)
