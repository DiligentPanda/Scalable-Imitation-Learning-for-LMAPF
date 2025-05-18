# for warehouse_large.map

import numpy as np

def gen_weights_warehouse_virtual_robots_001():
    weights=np.ones((20,23,5),dtype=np.int32)*2

    c=0
    for row in range(2,18,3):
        if (row-2)%6==0:
            # go east
            weights[row,3:20,0]=2-c
            # go west
            weights[row,3:20,2]=100000
        elif (row-2)%6==3:
            # go east
            weights[row,3:20,0]=100000
            # go west
            weights[row,3:20,2]=2-c
        else:
            assert False,row
            
    with open("warehouse_virtual_robots_w001.w","w") as f:
        f.write("[")
        for row in range(20):
            for col in range(23):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(19,22,4):
                        f.write(",")
        f.write("]")
    
gen_weights_warehouse_virtual_robots_001()