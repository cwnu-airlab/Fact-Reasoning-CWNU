import json
import torch
import transformers

torch.manual_seed(42)


class Service:
    task = [{'name': "text-summarization",'description': 'dummy system'}]

    def __init__(self):
        self.model = Model()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.model.run(content)
            status = 400 if 'error' in ret.keys() else 200
            return json.dumps(ret), status
        except Exception as e:
            return json.dumps({'error': "{}".format(e)}), 400

class Model():

    def __init__(self):


        config_dict = {
                "en;common-sense":"checkpoints/en_common-sense_t5-small_pgn_ms256_mt64/model",
                "kr;common-sense":"checkpoints/kr_common-sense_bert_ms128",
                "kr;law":"checkpoints/kr_legal_bert_ms128",
                }

        print('LOAD model', flush=True)
        archi_dict = dict()
        archi_dict['en;common-sense'] = {
                "tokenizer": transformers.T5Tokenizer,
                "model": transformers.T5ForConditionalGeneration,
                }
        archi_dict['kr;common-sense'] = {
                "tokenizer": transformers.BertTokenizer,
                "model": transformers.BertForSequenceClassification,
                }
        archi_dict['kr;law'] = {
                "tokenizer": transformers.BertTokenizer,
                "model": transformers.BertForSequenceClassification,
                }
        
        self.system_dict = dict()
        for key in archi_dict:
            if key not in config_dict: continue
            if not config_dict[key]: continue
            self.system_dict[key] = dict()
            self.system_dict[key]['model'] = archi_dict[key]['model'].from_pretrained(config_dict[key])
            if torch.cuda.is_available():
                self.system_dict[key]['model'] = self.system_dict[key]['model'].to('cuda')
            if 'en;common-sense' == key:
                self.system_dict[key]['tokenizer'] = archi_dict[key]['tokenizer'].from_pretrained('t5-small')
            else:
                self.system_dict[key]['tokenizer'] = archi_dict[key]['tokenizer'].from_pretrained('klue/bert-base')
            
    
    def get_ids(self, tokenizer, sentence):
        input_ids = tokenizer.encode(sentence)
        input_ids = torch.tensor([input_ids])
        return input_ids

    def get_one_supporting_facts(self, key, inputs):
        tokenizer = self.system_dict[key]['tokenizer']
        model = self.system_dict[key]['model']

        question = inputs['question']['text']
        context = inputs['context']
        form = "question: {} context: {}"
        
        max_score = -999
        max_score_sentence = ''
        for cont in context: 
            source = form.format(question, cont)
            source = self.get_ids(tokenizer, source)
            source = source.to(model.device)
            
            predict = model.forward(input_ids=source)
            predict = predict['logits']
            predict = torch.softmax(predict, dim=-1)
            predict = predict[0][1].item()
            if max_score < predict:
                max_source = predict
                max_score_sentence = cont
            
        output = max_score_sentence
        return output

    def get_supporting_facts(self, key, inputs):
        tokenizer = self.system_dict[key]['tokenizer']
        model = self.system_dict[key]['model']

        question = inputs['question']['text']
        context = inputs['context']
        form = "question: {} context: {}"

        output = list()
        for cont in context: 
            source = form.format(question, cont)
            source = self.get_ids(tokenizer, source)
            source = source.to(model.device)
            
            predict = model.forward(input_ids=source)
            predict = predict['logits']
            predict = torch.argmax(predict, dim=-1)
            predict = predict[0].item()
            if predict == 1:
                output.append(cont)
        if output == []:
            output = [self.get_one_supporting_facts(key, inputs)]
        print(f'PREDICT:{output}')
        return output


    def run(self, content):
        q_info = content['question']
        key = f"{q_info['language']};{q_info['domain']}"
        inputs = f"question: {q_info['text']} context: {content['context']}"

        if not q_info['text'] or not content['context']:
            return {'error':'invalid query'}
        else:
            output = self.get_supporting_facts(key, content)
            return {'output':output}

if __name__=='__main__':
    from nltk.tokenize import sent_tokenize
    import re
#     context_en = 'The Élysée Palace has been the official residence of the President of France since 1848. Dating to the early 18th century, it contains the office of the President and the meeting place of the Council of Ministers. It is located near the Champs-Élysées in the 8th arrondissement of Paris, the name Élysée deriving from Elysian Fields, the place of the blessed dead in Greek mythology.'
#     data = {'question': {'text':'in what city was the treaty signed by President Charles de Gaulle and Chancellor Konrad Adenauer signed?', 'domain': 'common-sense', 'language': 'en'}, 'context':context_en}
    context_kr = '엘리제 궁전은 1848년부터 프랑스 대통령의 관저이다. 18세기 초로 거슬러 올라가 대통령 집무실과 각료회의 장소가 담겨 있다. 파리 8구 샹젤리제 근처에 있는데, 그리스 신화에 나오는 축복받은 망자의 장소인 엘리시앙필드에서 유래한 엘리제라는 이름이다.'
    context_kr = sent_tokenize(context_kr)
    data = {'question': {'text':'샤를 드골 대통령과 콘라드 아데나워 총리가 서명한 조약은 어느 도시에서 이루어졌는가?', 'domain': 'common-sense', 'language': 'kr'}, 'context':context_kr}

    model = Model()

    question= data['question']
    key = f"{question['language']};{question['domain']}"
    print()
    print(f"TASK: {key}", flush=True)
    print(f'INPUT:\n{data}', flush=True)
#     contexts = [d['text'] for d in data['context']]
#     contexts += [' '.join(contexts)]
    result = list()
    for context in data['context']:
        context = re.sub('\([^\)]*\)','',context)
        context = re.sub('  *', ' ', context)
        context = sent_tokenize(context)
        data = {'question':question, 'context':context}
        predict = model.run(content = data)
        result.append(' '.join(predict['output']))
    print()
    print(f'RESULT:\n{result}')

    
