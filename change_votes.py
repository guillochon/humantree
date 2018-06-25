import json

with open('votes.json', 'r') as f:
    data = json.load(f)

new_data = {}

for k, v in data.items():
    if len(k) == 5:
        new_data['t' + k] = v
    else:
        new_data['q' + k] = v

with open('votes.json', 'w') as f:
    json.dump(new_data, f)
