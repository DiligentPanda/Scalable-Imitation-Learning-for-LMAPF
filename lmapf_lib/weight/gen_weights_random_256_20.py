# for warehouse_large.map

import numpy as np

def gen_weights_random_001_small():
    weights=np.ones((64,64,5),dtype=np.int32)*2

    c=1
    
    #: we need to add some consistent checking for weight assignment...
    #: we need a tool to visualize weight map...
    
    # : probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,64,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=2+c
        elif col%2==0:
            # go south
            weights[:,col,1]=2+c
            # go north
            weights[:,col,3]=2-c
    
    #### left ####
    for row in range(0,64,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=2+c
        elif row%2==0:
            # go east
            weights[row,:,0]=2+c
            # go west
            weights[row,:,2]=2-c
            
    with open("random_256_20_small_w001.w","w") as f:
        f.write("[")
        for row in range(64):
            for col in range(64):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(63,63,4):
                        f.write(",")
        f.write("]")        

def gen_weights_random_001_large():
    weights=np.ones((256,256,5),dtype=np.int32)*2

    c=1
    
    #: we need to add some consistent checking for weight assignment...
    #: we need a tool to visualize weight map...
    
    # : probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=2+c
        elif col%2==0:
            # go south
            weights[:,col,1]=2+c
            # go north
            weights[:,col,3]=2-c
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=2+c
        elif row%2==0:
            # go east
            weights[row,:,0]=2+c
            # go west
            weights[row,:,2]=2-c
            
    with open("random_256_20_large_w001.w","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")        



gen_weights_random_001_small()
gen_weights_random_001_large()