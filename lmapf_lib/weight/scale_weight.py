import json

with open("pibt_random_cma-es_piu-transfer_32x32_400_agents_four-way-move.w") as f:
    data = json.load(f)
    print(data)
    
for i in range(len(data)):
    data[i] = data[i]*100000
    
with open("pibt_random_cma-es_piu-transfer_32x32_400_agents_four-way-move.w","w") as f:
    json.dump(data,f)