import numpy as np
import json
from map import Map


def get_manhattan_dist(loc1, loc2, cols):
    return abs(loc1 % cols - loc2 % cols) + abs(loc1 // cols - loc2 // cols)


def uncompress_weights(map:Map,compressed_weights:np.ndarray, with_wait):
    MAX_WEIGHT=100000
    full_weights=np.ones(map.height*map.width*5,dtype=float)*MAX_WEIGHT
    
    dirs=[1,-map.width,-1,map.width]
    map_weights_idxs=[0,3,2,1]
    
    j=0
    if (with_wait):
        for i in range(map.height*map.width):
            x=i%map.width
            y=i//map.width
            if map.graph[y,x]==1:
                continue
            full_weights[i*5+4] = compressed_weights[j]
            j+=1
    else:
        j+=1 # the 0 indexed weight is for wait
    
    for i in range(map.width*map.height):
        x=i%map.width
        y=i//map.width
        
        if map.graph[y,x]==1:
            continue
        
        if not with_wait:
            full_weights[i*5+4] = compressed_weights[0]
        
        for d in range(4):
            dir=dirs[d]
            if 0<=i+dir and i+dir<map.width*map.height and \
                get_manhattan_dist(i,i+dir,map.width)<=1 and \
                    map.graph[(i+dir)//map.width,(i+dir)%map.width]==0:
                        weight=compressed_weights[j]
                        if weight==-1:
                            weight=MAX_WEIGHT
                        
                        map_weights_idx=map_weights_idxs[d]
                        full_weights[i*5+map_weights_idx]=weight
                        j+=1
    
    if j!=compressed_weights.size:
        print("Error: size mismatch, ", compressed_weights.size, " vs ", j)
        
    return full_weights

compressed_weight_path="../scripts/pibt_random_800_agents2_2000_steps.json"
map_path="../example_problems/random.domain/maps/random-32-32-20.map"
output_path="random_800_uncompressed_weights_with_wait_2000_steps.json"
map=Map(map_path)
map.print_graph(map.graph)

import json
with open(compressed_weight_path) as f:
    compressed_weights_json=json.load(f)

compressed_weights=compressed_weights_json["weights"]

compressed_weights=np.array(compressed_weights)

print(compressed_weights.min(),compressed_weights.max())
     
full_weights=uncompress_weights(map,compressed_weights,with_wait=True)

with open(output_path,"w") as f:
    json.dump(full_weights.tolist(),f)