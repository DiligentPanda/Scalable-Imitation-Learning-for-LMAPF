# for paris_1_256.map

import numpy as np
        
def gen_weights_brc202d_001():
    h=481
    w=530
    
    
    weights=np.ones((h,w,5),dtype=np.int32)*2

    c=0
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,w,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
    
    #### left ####
    for row in range(0,h,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
            
    with open("brc202d_weight_001.weight","w") as f:
        f.write("[")
        for row in range(h):
            for col in range(w):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(h-1,w-1,4):
                        f.write(",")
        f.write("]")        

def gen_weights_brc202d_002():
    h=481
    w=530
    
    weights=np.ones((h,w,5),dtype=np.int32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,w,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
    
    #### left ####
    for row in range(0,h,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
    
    
    ### try fix the congestion area
    # go east        
    weights[87:89,:,0]=100000
    # go west
    weights[87:89,:,2]=2-c
    
    # go east        
    weights[89:91,:,0]=2-c
    # go west
    weights[89:91,:,2]=100000
    
    # go south
    weights[:,420:426,1]=2-c
    # go north
    weights[:,420:426,3]=100000
    
    # go south
    weights[:,426:432,1]=100000
    # go north
    weights[:,426:432,3]=2-c
            
    with open("brc202d_weight_002.weight","w") as f:
        f.write("[")
        for row in range(h):
            for col in range(w):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(h-1,w-1,4):
                        f.write(",")
        f.write("]")    

def gen_weights_brc202d_003():
    h=481
    w=530
    
    weights=np.ones((h,w,5),dtype=np.int32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,w,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
    
    #### left ####
    for row in range(0,h,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
    
    
    ### try fix the congestion area
    # go east        
    weights[87:89,:,0]=100000
    # go west
    weights[87:89,:,2]=2-c
    
    # go east        
    weights[89:91,:,0]=2-c
    # go west
    weights[89:91,:,2]=100000
    
    # go south
    weights[:,420:426,1]=2-c
    # go north
    weights[:,420:426,3]=100000
    
    # go south
    weights[:,426:432,1]=100000
    # go north
    weights[:,426:432,3]=2-c
    
    
    # try fix the congestion area 2
    # go east        
    weights[132:137,:,0]=100000
    # go west
    weights[132:137,:,2]=2-c
    # go east        
    weights[137:142,:,0]=2-c
    # go west
    weights[137:142,:,2]=100000
    

    # go south
    weights[:,468:473,1]=2-c
    # go north
    weights[:,468:473,3]=100000
    
    # go south
    weights[:,473:478,1]=100000
    # go north
    weights[:,473:478,3]=2-c
            
    with open("brc202d_weight_003.weight","w") as f:
        f.write("[")
        for row in range(h):
            for col in range(w):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(h-1,w-1,4):
                        f.write(",")
        f.write("]")    






gen_weights_brc202d_003()