import re
import json
import requests

import hotpotqa_kr
import legal_kr

system = {
        'kr;common-sense': hotpotqa_kr.System(),
        'kr;law': legal_kr.System()
        }

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)


SPACE = '\t'

import time
def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


class Service():
    task = [
        {
            'name': "text-summarization",
            'description': 'dummy system'
        }
    ]

    def __init__(self):
        self.headers = {'Content-Type': 'application/json; charset=utf-8'} # optional
        self.sample_questions = [
                '데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?',
                '샤이닝을 부른 그룹 보컬의 고향은?',
                '철관 도매업용 토지에 대한 유휴토지에 해당여부',
                ]

    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        logging.info(content)
        if content['question'].strip() == '': content['question'] = content['example']
        if content['model'] == 'legislation': content['model'] = 'law' 
        #ret = self.predict(content)#; exit() ## XXX for dev
        try:
            ret = self.predict(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            logging.error(f"[func do] {e}")
            return json.dumps({'error': "{}".format(e)}), 400


    def predict(self, inputs):
        key = f"{inputs['language']};{inputs['model']}"
        logging.info(f'TASK:{key}')
        result = system[key].run(inputs)
        system_output = result.pop('results')

        system_output['final_output'] = list()
        #for i in range(len(result['answer'])):
        question = result['question']
        system_output['final_output'] += [f"<b>Question:</b></br>{question}</br>"]
        for i in range(3):
            answer = result['answer'][i]
            sp = result['sp'][i]
            score = result['score'][i]
            #system_output['final_output'] += [f"<b>A:</b>{answer} ({score*100:.2f})", f"<p style='background-color:#F2F2F2;'>{sp}</p>\n"]
            system_output['final_output'] += [f"<b>Answer:</b></br>{answer} ({score*100:.2f})", f"<b>Supporting Fact:</b></br>{sp}\n"]
        return {'output': get_result_text(system_output)}


    def sample(self,text):
        if text == '데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?':
            return [
                    '･Answer: 아니요, 다른 사람입니다.',
                    '･Supporting Facts:',
                    '\t･[<a href="https://ko.wikipedia.org/wiki/데드풀_(영화)"> 데드풀_(영화)] </a>]',
                    '\t\t시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다.',
                    '\t･[<a href="https://ko.wikipedia.org/wiki/킬러의_보디가드"> 킬러의 보디가드 </a>]',
                    '\t\t패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.'
                    ]
        elif text == '샤이닝을 부른 그룹 보컬의 고향은?':
            return [
                    '･ Answer: 서울',
                    '･ Supporting Facts:',
                    '\t･ [<a href="https://ko.wikipedia.org/wiki/자우림"> 자우림 </a>]',
                    '\t\t자우림(紫雨林)은 대한민국의 3인조 혼성 록 밴드이다. 기타를 맡은 이선규와 보컬의 김윤아, 베이스 기타의 김진만으로 구성되어 있으며, 드럼의 구태훈은 탈퇴하였다.',
                    '\t･ [<a href="https://ko.wikipedia.org/wiki/김윤아"> 김윤아 </a>]',
                    '\t\t김윤아는 대한민국 서울특별시 강남구에서 태어났다.',
                    '\t\t(김윤아, 출생, 대한민국 서울특별시 강남구)'
                    ]
        else:
            raise KeyError('"{}" is not in sample list.'.format(text))

def get_result_text(system_result):

    count = 0
    result = []
    result += ['\n'.join(system_result['final_output']).replace('\t','&nbsp;'*4),'\n\n']
    result += ['<details style="border:1px solid #aaa;border-radius:4px" open="open">\n']
    result += ['<summary style="text-align:center; font-weight:bold"> Real Result of System </summary>']
    for key in [3,1,4,5,2,7,9,11,8]:
        if key not in system_result: continue
        system_result[key]['input'] = json.dumps(system_result[key]['input'], indent=4, ensure_ascii=False)
        system_result[key]['input'] = system_result[key]['input'].replace(' ','&nbsp;')
        if 'error' in system_result[key]['output'] and system_result[key]['output']['error']:
            title = '<summary>'+html_font(f"<{key}_{system_result[key]['name']}-{system_result[key]['manager']}>",cls='error_comment',color='#FAC3C3')+'</summary>'
            system_result[key]['output'] = html_font(system_result[key]['output']['error'],cls='error_comment',color='#FFFFFF',font_color='red')
            output = f"<details open='open' style='border:1px solid #aaa;border-radius:4px'> <summary>Output</summary> {system_result[key]['output']}</details>"
        else:
            title = '<summary>'+html_font(f"<{key}_{system_result[key]['name']}-{system_result[key]['manager']}>")+'</summary>'
            system_result[key]['output'] = json.dumps(system_result[key]['output'], indent=4, ensure_ascii=False)
            system_result[key]['output'] = system_result[key]['output'].replace(' ','&nbsp;')
            output = f"<details style='border:1px solid #aaa;border-radius:4px'> <summary>Output</summary> {system_result[key]['output']}</details>"
        text = ['<details>',
                title,
                f"<details style='border:1px solid #aaa;border-radius:4px'> <summary>Input</summary> {system_result[key]['input']}</details> ",
                output,
                '</details>']
        result += '\n'.join(text)+'\n'
    result += ['</details>\n']
    result = ''.join(result)
    return result.replace('\n','<br/>')


def html_font(text,cls='result_head',color='powderblue',font_color='#000000'):
    return f'<a class="{cls}" style="background-color:{color};font-weight:bold;color:{font_color}">'+text+'</a>'


if __name__=='__main__':
    import sys
    service = Service()
    content = {'question': '샤를 드골 대통령과 콘라드 아데나워 총리가 서명한 조약은 어느 도시에서 이루어졌는가?', 
            'cli_ip': '10.100.54.146', 'model': 'common-sense', 'example': 'None', 'language': 'kr'}
    predict = service.do(content = content)
