import json
import numpy as np
import matplotlib.pyplot as plt

snapshot_step=5000
input_fp='test_400_w021.json'
output_fn=input_fp.replace(".json","_wait_map.png")
wait_heatmap_fn=input_fp.replace(".json","_wait_heatmap.json")
h=32
w=32

def save_wait_heatmap(wait_heatmap, wait_heatmap_fn):
    import json
    with open(wait_heatmap_fn,'w') as f:
        json.dump([int(v) for v in wait_heatmap.flatten()],f)

with open(input_fp) as f:
    data = json.load(f)

print(data.keys())
print(data["actionModel"])

# print(data["start"])


def get_orient_idx(orient):
    return {"E": 0, 'S': 1, 'W': 2, 'N': 3}[orient]

def move(x,y,orient,action):
    if action=="F":
        if orient==0:
            x+=1
        elif orient==1:
            y+=1
        elif orient==2:
            x-=1
        elif orient==3:
            y-=1
    elif action=="R":
        orient=(orient+1)%4
    elif action=="C":
        orient=(orient-1)%4
    elif action=="W":
        pass
    return x,y,orient


team_size=data["teamSize"]
actual_paths=data["actualPaths"]
starts=data["start"]
events=data["events"]
tasks=data["tasks"]


MAXT=(len(actual_paths[0])+1)//2

T=MAXT

arr=np.zeros((team_size,T+1,3),dtype=int)

goal_locs=np.zeros((team_size,),dtype=int)

wait_heatmap=np.zeros((h,w),dtype=int)

for aid in range(team_size):
    start=starts[aid]
    arr[aid,0,0]=start[1]
    arr[aid,0,1]=start[0]
    arr[aid,0,2]=get_orient_idx(start[2])
    
    path=actual_paths[aid]
    actions=path.split(",")
    x=arr[aid,0,0]
    y=arr[aid,0,1]
    orient=arr[aid,0,2]
    for t in range(T):
        action=actions[t]
        if action=="W":
            wait_heatmap[y,x]+=1
        x,y,orient=move(x,y,orient,action)
        arr[aid,t+1,0]=x
        arr[aid,t+1,1]=y
        arr[aid,t+1,2]=orient
        
print(wait_heatmap.max(),wait_heatmap.max()/MAXT)

save_wait_heatmap(wait_heatmap,wait_heatmap_fn)

plt.imshow(wait_heatmap)


plt.colorbar()
plt.tight_layout()
plt.savefig(output_fn)
plt.show()
