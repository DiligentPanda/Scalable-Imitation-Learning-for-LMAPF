import json
import numpy as np
import matplotlib.pyplot as plt

from light_malib.envs.LMAPF.map import Map

import matplotlib.cm as cm
import matplotlib.colors as colors

from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({'font.size': 26})
fig, axes = plt.subplots(1,1,figsize=(12,4))
fig.tight_layout(h_pad=0, w_pad=0)
fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95)

#c_map = cm.get_cmap('Reds')

c_map = colors.LinearSegmentedColormap.from_list('nameofcolormap',['w','r'],gamma=0.5)

c_map.set_bad('black')

# input_fp=r'1207\1207\test_warehouse_lacam2_wno.json'
# output_fn="figs/wait_heatmap_warehouse_lacam2_wno_2500_steps.gif"
input_fp=    r'logs/main_reeval_weighted/v3/RL/warehouse/best/eval/2024-10-09-12-13-01_warehouse_small_uniform_PIBT-RL_600/log/log_0.json'
output_fn="figs/wait_heatmap_warehouse_small_uniform_500_steps.gif"
map_fp=r"lmapf_lib/data/warehouse_small.map"
# output_fn=input_fp.replace(".json","_wait_map.pdf")
wait_heatmap_fn=input_fp.replace(".json","_wait_heatmap.json")


m=Map(map_fp)
h=m.height
w=m.width


def save_wait_heatmap(wait_heatmap, wait_heatmap_fn):
    import json
    with open(wait_heatmap_fn,'w') as f:
        json.dump([int(v) for v in wait_heatmap.flatten()],f)

with open(input_fp) as f:
    data = json.load(f)

print(data.keys())
print(data["actionModel"])

print(data["numTaskFinished"])

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

T=500#MAXT

arr=np.zeros((team_size,T+1,3),dtype=int)

goal_locs=np.zeros((team_size,),dtype=int)

wait_heatmap=np.zeros((T,h,w),dtype=float)

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
            wait_heatmap[t,y,x]+=1
        x,y,orient=move(x,y,orient,action)
        arr[aid,t+1,0]=x
        arr[aid,t+1,1]=y
        arr[aid,t+1,2]=orient
    

wait_heatmap=np.cumsum(wait_heatmap,axis=0)

print(wait_heatmap.max(),wait_heatmap.max()/MAXT)


elapse=10
num_frames=T//elapse

def animate(t):
    if t>=num_frames:
        t=num_frames-1
    t=t*elapse
    print(t)
    axes.clear()
    _wait_heatmap=wait_heatmap[t]/(t+1)
    _wait_heatmap[m.graph==1] = np.nan
    axes.imshow(_wait_heatmap,cmap=c_map,interpolation='none',vmin=0,vmax=1)
    axes.axis('off')
    axes.set_title("T={:04d}".format(t))
    return axes

animate(0)

ani = FuncAnimation(fig, animate, frames=num_frames+10, repeat=True)


# ax=axes
# wait_heatmap=wait_heatmap.astype(float)/T
# wait_heatmap[m.graph==1] = np.nan
# im = ax.imshow(wait_heatmap,cmap=c_map,interpolation='none',vmin=0,vmax=1)
# ax.axis('off')

# ax.set_title(xlabels[idx], y=-0.25)

# fig.subplots_adjust(right=0.9, wspace=0.025, hspace=None)
# cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
# cbar=fig.colorbar(im, cax=cbar_ax)
# print(cbar.get_ticks())
# cbar.set_ticklabels([(">" if idx==len(cbar.get_ticks())-1 else "") +str(int(v)) for idx,v in enumerate(cbar.get_ticks())])

# plt.axis('off')
# plt.colorbar()
# plt.savefig(output_fn,  pad_inches = 0, transparent=True, bbox_inches = 'tight')
ani.save(output_fn, dpi=300, writer=PillowWriter(fps=10))
#plt.show()
