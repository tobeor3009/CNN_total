import json

with open("./config.json") as json_text:
    CONFIG = json.load(json_text)
    print(CONFIG)
