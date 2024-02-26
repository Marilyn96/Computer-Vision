import requests
from PIL import Image
import base64
import json
from base64 import b64encode


class Base64Encoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return json.JSONEncoder.default(self, o)

"""
URL = "https://api.teatscoring.tech/api/CowUdderImages"
with open('C:/Users/test/Documents/Academics/2021/EPR402/Final Report/Picture6.png', "rb") as img_file:
"""


def sendImage(address, scores):
    URL = "https://api.teatscoring.tech/api/CowUdderImages"
    with open(address, "rb") as img_file:
        string64 = base64.b64encode(img_file.read()).decode('utf-8')
    PARAMS = {"avgScore": scores[0],
              "score1": scores[1],
              "score2": scores[2],
              "score3": scores[3],
              "score4": scores[4],
              "imageData": string64}

    r = requests.post(url=URL, json=PARAMS)
    data=r.json()
    print(data)



