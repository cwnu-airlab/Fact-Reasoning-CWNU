import json
import requests
from urllib.parse import urljoin

URL = 'http://thor.nlp.wo.tc:12341/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

task_q = '/api/task'
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

context_kr = '엘리제 궁전은 1848년부터 프랑스 대통령의 관저이다. 18세기 초로 거슬러 올라가 대통령 집무실과 각료회의 장소가 담겨 있다. 파리 8구 샹젤리제 근처에 있는데, 그리스 신화에 나오는 축복받은 망자의 장소인 엘리시앙필드에서 유래한 엘리제라는 이름이다.'
data = {'question': {'text':'샤를 드골 대통령과 콘라드 아데나워 총리가 서명한 조약은 어느 도시에서 이루어졌는가?', 'model': 'common-sense', 'language': 'kr'}, 'context':context_kr}

data = json.dumps(data)
response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.json())


context_en = 'The Élysée Palace has been the official residence of the President of France since 1848. Dating to the early 18th century, it contains the office of the President and the meeting place of the Council of Ministers. It is located near the Champs-Élysées in the 8th arrondissement of Paris, the name Élysée deriving from Elysian Fields, the place of the blessed dead in Greek mythology.'
data = {'question': {'text':'in what city was the treaty signed by President Charles de Gaulle and Chancellor Konrad Adenauer signed?', 'model': 'common-sense', 'language': 'en'}, 'context':context_en}

data = json.dumps(data)
response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.json())
