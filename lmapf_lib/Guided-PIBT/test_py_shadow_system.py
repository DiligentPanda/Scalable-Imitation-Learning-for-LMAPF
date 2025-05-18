import sys
sys.path.insert(0,"guided-pibt-build")

import py_shadow_system
import numpy as np

map_path="/root/GRF_MARL/lmapf_lib/data/random-32-32-20.map"
agents_path="/root/GRF_MARL/lmapf_lib/Guided-PIBT/Benchmark-Archive/2023 Competition/Main Round Evaluation Instances/random.domain/agents/random-32-32-20_400.agents"
tasks_path="/root/GRF_MARL/lmapf_lib/Guided-PIBT/Benchmark-Archive/2023 Competition/Main Round Evaluation Instances/random.domain/tasks/random-32-32-20_400.tasks"

with open(agents_path) as f:
    agents=[]
    # skip the first line
    f.readline()
    for line in f:
        agents.append(int(line.strip()))
        
with open(tasks_path) as f:
    tasks=[]
    # skip the first line
    f.readline()
    for line in f:
        tasks.append(int(line.strip()))

system=py_shadow_system.PyShadowSystem(map_path)
for iter in range(1):
    system.reset(agents,tasks,209652396)
    locations=system.query_locations()
    locations=np.array(locations)
    y=locations//32
    x=locations%32
    print(y[:5],x[:5])
    goals=system.query_goals()
    goals=np.array(goals)
    gy=goals//32
    gx=goals%32
    print(gy[:5],gx[:5])
    for i in range(320):
        # locations=system.query_locations()
        # print(locations)
        # views=np.arange(-60,61,dtype=int).tolist()
        # heuristics=system.query_heuristics(locations,11,11)
        # heuristics=np.array(heuristics,dtype=np.float32)
        # heuristics=heuristics.reshape(len(locations),11,11)
        # print(heuristics.shape)
        
        # print("iter",i)
        # import time
        # time.sleep(1)
        actions=system.query_pibt_actions()
        # print(actions)
        system.step(actions)
        # print("next step")
        # import time
        # time.sleep(1)
    
    import time
    time.sleep(1)
