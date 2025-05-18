import json
import numpy as np
import matplotlib.pyplot as plt

from light_malib.envs.LMAPF.map import Map

import matplotlib.cm as cm
import matplotlib.colors as colors

plt.rcParams.update({'font.size': 26})
fig, axes = plt.subplots(1,4,figsize=(32,4))
fig.tight_layout(h_pad=0, w_pad=0)

#c_map = cm.get_cmap('Reds')

c_map = colors.LinearSegmentedColormap.from_list('nameofcolormap',['w','r'],gamma=0.5)

c_map.set_bad('black')

xlabels=["PIBT without Guidance Graph", "WPPL without Guidance Graph", "PIBT with Guidance Graph", "WPPL with Guidance Graph"]

output_fn="figs/wait_heatmap_warehouse_small.pdf"
for idx, input_fp in enumerate([
    r'logs/main_reeval_weighted/v3/RL/warehouse/best/eval/2024-10-09-12-13-01_warehouse_small_uniform_PIBT-RL_600/log/log_0.json',
    r'logs/main_reeval_weighted/v3/RL/warehouse/best/eval/2024-10-09-12-13-01_warehouse_small_uniform_PIBT-RL_600/log/log_1.json',
    r'logs/main_reeval_weighted/v3/RL/warehouse/best/eval/2024-10-09-12-13-01_warehouse_small_uniform_PIBT-RL_600/log/log_2.json',
    r'logs/main_reeval_weighted/v3/RL/warehouse/best/eval/2024-10-09-12-13-01_warehouse_small_uniform_PIBT-RL_600/log/log_3.json'
]):
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

    wait_heatmap=wait_heatmap.astype(float)
    wait_heatmap[m.graph==1] = np.nan
    im = axes[idx].imshow(wait_heatmap,cmap=c_map,interpolation='none',vmin=0,vmax=MAXT)
    axes[idx].axis('off')
    
    axes[idx].set_title(xlabels[idx], y=-0.25)

fig.subplots_adjust(right=0.9, wspace=0.025, hspace=None)
cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
cbar=fig.colorbar(im, cax=cbar_ax)
# print(cbar.get_ticks())
# cbar.set_ticklabels([(">" if idx==len(cbar.get_ticks())-1 else "") +str(int(v)) for idx,v in enumerate(cbar.get_ticks())])

# plt.axis('off')
# plt.colorbar()
plt.savefig(output_fn,  pad_inches = 0, transparent=True, bbox_inches = 'tight')
#plt.show()
