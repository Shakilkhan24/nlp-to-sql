import json 
with open('sample.json','r') as f:
    data=json.load(f)

print(data['question'])
print(data['table']['header'])
print('_______________________________')
print(data['table']['rows'])    