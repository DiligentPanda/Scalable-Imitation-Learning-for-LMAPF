# for warehouse_large.map

import numpy as np

def gen_weights_warehouse_small_001():
    weights=np.ones((33,57,5),dtype=np.int32)*2

    c=0
    for row in range(7,26,3):
        if (row-7)%6==0:
            # go east
            weights[row,7:47,0]=2-c
            # go west
            weights[row,7:47,2]=100000
        elif (row-7)%6==3:
            # go east
            weights[row,7:47,0]=100000
            # go west
            weights[row,7:47,2]=2-c
        else:
            assert False,row
            
    for col in range(7,47,4):
        if (col-7)%8==0:
            # go south
            weights[7:26,col,1]=2-c
            # go north
            weights[7:26,col,3]=100000
        elif (col-7)%8==4:
            # go south
            weights[7:26,col,1]=100000
            # go north
            weights[7:26,col,3]=2-c
        else:
            assert False
            
    with open("warehouse_small_w001.w","w") as f:
        f.write("[")
        for row in range(33):
            for col in range(57):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(32,56,4):
                        f.write(",")
        f.write("]")
    
def gen_weights_warehouse_large_001():
    weights=np.ones((140,500,5),dtype=np.int32)*2

    c=0
    for row in range(7,131,3):
        if (row-7)%6==0:
            # go east
            weights[row,7:492,0]=2-c
            # go west
            weights[row,7:492,2]=100000
        elif (row-7)%6==3:
            # go east
            weights[row,7:492,0]=100000
            # go west
            weights[row,7:492,2]=2-c
        else:
            assert False,row
            
    for col in range(7,492,4):
        if (col-7)%8==0:
            # go south
            weights[7:131,col,1]=2-c
            # go north
            weights[7:131,col,3]=100000
        elif (col-7)%8==4:
            # go south
            weights[7:131,col,1]=100000
            # go north
            weights[7:131,col,3]=2-c
        else:
            assert False
            
    with open("warehouse_large_w001.w","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")
    
    
gen_weights_warehouse_small_001()
gen_weights_warehouse_large_001()