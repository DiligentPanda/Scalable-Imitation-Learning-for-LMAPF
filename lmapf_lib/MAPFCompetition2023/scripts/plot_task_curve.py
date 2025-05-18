
import json
import numpy as np

input_fp='test_random_800_w010.json'
output_fp=input_fp.replace(".json","_task_curve.png")

with open(input_fp) as f:
    data = json.load(f)

print(data.keys())

events=data["events"]
team_size=data["teamSize"]
actual_paths=data["actualPaths"]
print("team size:", team_size)

MAXT=(len(actual_paths[0])+1)//2

tasks_completed_by_step=np.zeros(MAXT+1,dtype=int)
tasks_assigned_by_step=np.zeros(MAXT+1,dtype=int)

for agent_events in events:
    for event in agent_events:
        tid,timestep,info=event
        if info=="finished":
            tasks_completed_by_step[timestep]+=1
        elif info=="assigned":
            tasks_assigned_by_step[timestep]+=1
        else:
            raise NotImplementedError

print("total completed:", sum(tasks_completed_by_step))
print("total assigned:", sum(tasks_assigned_by_step))

import matplotlib.pyplot as plt

accumulated_tasks_assigned_by_step=np.cumsum(tasks_assigned_by_step)
accumulated_tasks_completed_by_step=np.cumsum(tasks_completed_by_step)


plt.plot(np.arange(len(accumulated_tasks_completed_by_step)),accumulated_tasks_completed_by_step,label="completed")

plt.show()

plt.tight_layout()

plt.savefig(output_fp)