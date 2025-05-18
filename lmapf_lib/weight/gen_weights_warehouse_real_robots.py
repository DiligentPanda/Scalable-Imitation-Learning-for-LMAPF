# for warehouse_large.map

import numpy as np

def gen_weights_warehouse_real_robots_001():
    weights=np.ones((12,15,5),dtype=np.int32)*2

    c=0
    for row in range(1,11,3):
        if (row-1)%6==0:
            # go east
            weights[row,2:13,0]=2-c
            # go west
            weights[row,2:13,2]=100000
        elif (row-1)%6==3:
            # go east
            weights[row,2:13,0]=100000
            # go west
            weights[row,2:13,2]=2-c
        else:
            assert False,row
            
    with open("warehouse_real_robots_w001.w","w") as f:
        f.write("[")
        for row in range(12):
            for col in range(15):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(11,14,4):
                        f.write(",")
        f.write("]")
    
gen_weights_warehouse_real_robots_001()