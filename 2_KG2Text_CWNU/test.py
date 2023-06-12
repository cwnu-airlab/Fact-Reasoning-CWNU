import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:12342/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional
with open('app/data/sample_legal_1.json','r') as f:
    data = json.load(f)
print('input:')
print(data)
data = json.dumps(data)

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())
