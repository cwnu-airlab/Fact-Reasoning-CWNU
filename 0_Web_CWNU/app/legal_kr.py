import re
import json
import requests
from urllib.parse import urljoin

from nltk.tokenize import sent_tokenize
from Espresso_ne.pyEspresso import Espresso
espresso = Espresso()

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)


import time
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        #print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

class System():

    def __init__(self):
        self.headers = {'Content-Type': 'application/json; charset=utf-8'} # optional
        self.url_dict = {
                3:'http://thor.nlp.wo.tc:12343',
                #3:'http://ketiair.com:10021',
                #4:'http://ketiair.com:10022',
                4:'http://thor.nlp.wo.tc:12344',
                1:'http://thor.nlp.wo.tc:12341',
                2:'http://thor.nlp.wo.tc:12342',
                11: 'http://thor.nlp.wo.tc:12311',
                8: 'http://thor.nlp.wo.tc:12348',
                }

        kb_size = 3000
        self.wiki_doc = dict()
        context_filename = 'data/total_context_legal_kr_triple_keti.jsonl'
        with open(context_filename,'r') as f:
            self.wiki_doc['kr'] = [json.loads(d) for d in f][:kb_size]
#         with open('data/total_context_en.jsonl','r') as f:
#             self.wiki_doc['en'] = [json.loads(d) for d in f][:kb_size]
        self.doc_dict = {d[0]:d for d in self.wiki_doc['kr']}
        self.wiki_doc['kr'] = [[d[0],d[1]] for d in self.wiki_doc['kr']]

    
    def run(self, inputs):
        system_result = dict()

        #logging.info(f"[0] {inputs}")
        question = {'text':inputs['question'], 'language':inputs['language'], 'domain':inputs['model']}
        #logging.info(f"[question] {question}")
        #print(f"\nINPUT:\n{question}\n", flush=True)

        #print("\n[3] Passage Retrieval:")
        system_result[3] = self.get_passage_retrieval(question)
        #print(f"\n{system_result[3]['output']}", flush=True)
        #logging.info(f"[3] {system_result[3]['output']}")

        triples = list()
        for item in system_result[3]['output']:
            title = item['title']
            triple = self.doc_dict[title][2]
            triples.append(triple)
        triples = [y for x in triples for y in x]
        
        #print("\n[4] Relation Extraction:")
        #system_result[4] = self.get_relation_extraction(question, system_result[3]['output'])
        system_result[4] = {'output':triples, 'input':{'question':question}, 'name':'RelationExtraction', 'manager':'KETI'}
        #print(f"\n{system_result[4]['output']}", flush=True)
        #logging.info(f"[4] {system_result[4]['output']}")

        #print("\n[1] Summarize:")
        system_result[1] = self.get_summarize(question, system_result[3]['output'])
        #logging.info(f"[1] {system_result[1]['output']}")
        #print(f"\n{system_result[1]['output']}", flush=True)

        #print("\n[2] KG-to-Text:")
        system_result[2] = self.get_kg2text(question, system_result[4]['output'])
        #logging.info(f"[2] {system_result[2]['output']}")
        #print(f"\n{system_result[2]['output']}", flush=True)
        
        sp_list = system_result[1]['output']
        sp_list += [[d] for d in system_result[2]['output']]
        sp_list += [[re.sub('[ \t\n]+',' ',d['text'].strip())] for d in system_result[3]['output']]
        sp_list = sp_list[1:]
        sp_list = [' '.join(d) for d in sp_list]
        sp_list = [d for d in sp_list if '아예 그 남기고 이미 잔여면' not in d]
        sp_list = [[d] for d in sp_list]
        
        #print("\n[11] Answering:")
        system_result[11] = self.get_answer(question, sp_list)
        #logging.info(f"[11] {system_result[11]['output']}")
        #print(f"\n{system_result[11]['output']}", flush=True)

        #print("\n[8] ReRanking:")
        system_result[8] = self.get_rerank(question, system_result[11]['output'], sp_list)
        #print(f"\n{system_result[8]['output']}", flush=True)
        #logging.info(f"[8] {system_result[8]['output']}")

        ## set result
        a_list = system_result[8]['input']['answer_list']
        sp_list = system_result[8]['input']['supporting_facts']
        score = system_result[8]['output']
        
        output = sorted(zip(score, a_list, sp_list), key=lambda x:x[0], reverse=True)
        output = [d for d in output if d[1] != '다.']
        score, answer, sp  = [list(d) for d in zip(*output)]
        
        result = {'question':question['text'], 'answer':answer, 'sp':sp, 'score':score}
        #print(f"\nRESULT:\n{result}", flush=True)
        result['results'] = system_result
        return result
        

    def get_output(self,url,data):
        data = json.dumps(data)
        try:
            response = requests.post(urljoin(url, '/api/task'), data=data, headers=self.headers)
            return response.json(), response.status_code
        except requests.exceptions.ConnectionError as e:
            logging.error(e)
            error = {'error':'{}'.format(e)}
            return error, 521


    def get_entity_pairs(self, sentence):
        ne = espresso.get_ne(sentence)
        result = list()
        for i,ent_i in enumerate(ne):
            for ent_j in ne[i+1:]:
                result.append([ [ent_i['begin'],ent_i['end']], [ent_j['begin'],ent_j['end']] ])
        return result, ne


    @logging_time
    def get_rerank(self, question, answer_list, sp_list):
        sp_list = [d[0] for d in sp_list]
        data = {'question':question, 'answer_list':answer_list, 'supporting_facts':sp_list}
        out, status = self.get_output(self.url_dict[8], data)
        score = out['reranking_score']
        if status == 200:
            out['output'] = [[d] for d in out['supporting_facts']]
        else:
            out['error'] = f"[{status}] {out['error']}"
        result = {'name':'ReRanking', 'manager':'POSTECH', 'input':data, 'output':score}
        return result

    
    @logging_time
    def get_answer(self, question, sp_list):
        output = {'output':list(), 'error':list()}
        for sp in sp_list:
            if type(sp) == list: sp = ' '.join(sp)
            data = {'question':question, 'context':sp.strip()}
            out, status = self.get_output(self.url_dict[11], data)
            if status == 200:
                answer = out['answer'][0].strip()
                if answer[-2:] != '다.':
                    answer = answer.strip().split('다.')[:-1]
                    answer = '다.'.join(answer)+'다.'
                output['output'].append(answer)
            else:
                output['error'].append(f"[{status}] {out['error']}")
        data = {'question':question, 'context':sp_list}
        result = {'name':'Answering', 'manager':'CWNU', 'input':data, 'output':output['output']}
        return result

    @logging_time
    def get_kg2text(self, question, triples):
        data = {'question':question, 'triples':triples}
        out, status = self.get_output(self.url_dict[2], data)
        if status != 200:
            out['error'] = f"[{status}]{out['error']}"
        result = {'name':'KG2Text', 'manager':'CWNU', 'input':data, 'output':out['output']}
        return result

    @logging_time
    def get_summarize(self, question, context_list):
        contexts = [d['text'] for d in context_list]
        contexts.append(' '.join(contexts))
        output = {'output':list(), 'error':list()}
        for context in contexts:
            context = re.sub('\([^\)]*\)','',context)
            context = re.sub('  *', ' ', context)
            context = sent_tokenize(context)
            data = {'question':question, 'context':context}
            out, status = self.get_output(self.url_dict[1], data)
            if status == 200:
                output['output'].append(out['output'])
            else:
                output['error'].append(f"[{status}] {out['error']}")
        data = {'question':question, 'context':context_list}
        result = {'name':'Summarizer', 'manager':'CWNU', 'input':data, 'output':output['output']}
        return result

    
    @logging_time
    def get_passage_retrieval(self, question, topk=2):
        data = {'question':question, 'context':self.wiki_doc[question['language']], 'max_num_retrieved':2}
        out, status = self.get_output(self.url_dict[3], data)
        out = out['retrieved_doc'][:topk]
        if status != 200:
            out['error'] = f"[{status}]{out['error']}"
        result = {'name':'PassageRetriever', 'manager':'KETI', 'input':data, 'output':out}
        return result


    @logging_time
    def get_relation_extraction(self, question):
        entity_pairs, ne_info = self.get_entity_pairs(question['text'])
        data = {'doc':question, 'arg_pairs':[d for d in entity_pairs], 'tagger':ne_info}
        out, status = self.get_output(self.url_dict[4], data)

        output = list()
        for index,rel in zip(entity_pairs, out['result']):
            text = question['text']
            subjects = text[index[0][0]:index[0][1]+1]
            objects = text[index[1][0]:index[1][1]+1]
            output.append([subjects, rel, objects])
        result = {'name':'RelationRetrieval', 'manager':'KETI', 'input':data, 'output':output}
        return result



if __name__=='__main__':
    import sys
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            inputs = json.load(f)
    else:
#         inputs = {
#                 'question': '철관 도매업용 토지에 대한 유휴토지에 해당여부', 
#                 'cli_ip': '10.100.54.146', 'model': 'law', 'example': 'None', 'language': 'kr'
#                 }
        inputs = {
                'question': '기간 내에 상속을 포기하지 못했어요.', 
                'cli_ip': '10.100.54.146', 'model': 'law', 'example': 'None', 'language': 'kr'
                }
#         inputs = {
#                 'question': '임차인이 토지를 무단으로 전대한 경우 임대인은 임차인과의 임대차 계약을 해지할 수 있나요.', 
#                 'cli_ip': '10.100.54.146', 'model': 'law', 'example': 'None', 'language': 'kr'
#                 }

    system = System()
    result = system.run(inputs)
    result.pop('results')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
