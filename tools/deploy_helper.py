import requests
import json
import time

url = "http://128.114.152.24:3000/query"

def query_system(prompt:str,site:str) -> str:
    time.sleep(10)
    payload = json.dumps({
      "site": site,
      "prompt": prompt
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if(response.status_code != 200):
        print(response.text)
        return "Error unable to fetch data"
    return response.json()['response']
