# for paris_1_256.map

import numpy as np
        
def gen_weights_paris_001():
    weights=np.ones((256,256,5),dtype=np.int32)*2

    c=0
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
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
    for row in range(0,256,1):
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
            
    with open("paris_weight_001.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")        

def gen_weights_paris_002():
    weights=np.ones((256,256,5),dtype=np.int32)*2

    c=0
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
            
    with open("paris_weight_002.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")       


def gen_weights_paris_010():
    weights=np.ones((256,256,5),dtype=np.int32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3
            
    with open("paris_weight_010.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")       

def gen_weights_paris_011():
    weights=np.ones((256,256,5),dtype=np.float32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3

    for row in range(88,98):
        for col in range(93,102):
            weights[row,col,:]+=0.1
            
    for row in range(155,165):
        for col in range(99,110):
            weights[row,col,:]+=0.1

    for row in range(67,81):
        for col in range(163,168):
            weights[row,col,:]+=0.1
            
    for row in range(18,24):
        for col in range(86,94):
            weights[row,col,:]+=0.1
            
    for row in range(187,216):
        for col in range(249,252):
            weights[row,col,:]+=0.1

    with open("paris_weight_011.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")    

def gen_weights_paris_012():
    weights=np.ones((256,256,5),dtype=np.float32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3

    for row in range(85,100):
        for col in range(90,105):
            weights[row,col,:]+=0.2
            
    for row in range(155,165):
        for col in range(99,110):
            weights[row,col,:]+=0.1

    for row in range(67,81):
        for col in range(163,168):
            weights[row,col,:]+=0.1
            
    for row in range(18,24):
        for col in range(86,94):
            weights[row,col,:]+=0.1
            
    for row in range(187,216):
        for col in range(249,252):
            weights[row,col,:]+=0.1
            
    for row in range(160,165):
        for col in range(157,162):
            weights[row,col,:]+=0.1

    with open("paris_weight_012.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")    


def gen_weights_paris_013():
    weights=np.ones((256,256,5),dtype=np.float32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3

    for row in range(85,100):
        for col in range(90,105):
            weights[row,col,:]+=0.2
            
    for row in range(155,165):
        for col in range(99,110):
            weights[row,col,:]+=0.1

    for row in range(67,81):
        for col in range(163,168):
            weights[row,col,:]+=0.1
            
    for row in range(18,24):
        for col in range(86,94):
            weights[row,col,:]+=0.1
            
    for row in range(187,216):
        for col in range(249,252):
            weights[row,col,:]+=0.1
            
    for row in range(160,165):
        for col in range(157,162):
            weights[row,col,:]+=0.1
            
    for row in range(188,194):
        for col in range(90,95):
            weights[row,col,:]+=0.05
    
    for row in range(140,147):
        for col in range(158,160):
            weights[row,col,:]+=0.05
            
    for row in range(118,123):
        for col in range(134,141):
            weights[row,col,:]+=0.05
            
    for row in range(49,59):
        for col in range(123,128):
            weights[row,col,:]+=0.05
            
    for row in range(94,102):
        for col in range(87,105):
            weights[row,col,:]+=0.05

    with open("paris_weight_013.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")    


def gen_weights_paris_014():
    weights=np.ones((256,256,5),dtype=np.float32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3

    for row in range(85,100):
        for col in range(90,105):
            weights[row,col,:]+=0.2
            
    for row in range(155,165):
        for col in range(99,110):
            weights[row,col,:]+=0.1

    for row in range(67,81):
        for col in range(163,168):
            weights[row,col,:]+=0.1
            
    for row in range(18,24):
        for col in range(86,94):
            weights[row,col,:]+=0.1
            
    for row in range(187,216):
        for col in range(249,252):
            weights[row,col,:]+=0.1
            
    for row in range(160,165):
        for col in range(157,162):
            weights[row,col,:]+=0.1
            
    for row in range(188,194):
        for col in range(90,95):
            weights[row,col,:]+=0.05
    
    for row in range(140,147):
        for col in range(158,160):
            weights[row,col,:]+=0.05
            
    for row in range(118,123):
        for col in range(134,141):
            weights[row,col,:]+=0.05
            
    for row in range(49,59):
        for col in range(123,128):
            weights[row,col,:]+=0.05
            
    for row in range(84,102):
        for col in range(87,106):
            weights[row,col,:]+=0.1

    with open("paris_weight_014.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")    

def gen_weights_paris_015():
    weights=np.ones((256,256,5),dtype=np.float32)*2

    c=1
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,256,1): 
        # any
        if col%2==1:
            # go south
            weights[:,col,1]=3
            # go north
            weights[:,col,3]=2-c
        elif col%2==0:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=3
    
    #### left ####
    for row in range(0,256,1):
        # any
        if row%2==1:
            # go east
            weights[row,:,0]=3
            # go west
            weights[row,:,2]=2-c
        elif row%2==0:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=3

    for row in range(85,100):
        for col in range(90,105):
            weights[row,col,:]+=0.2
            
    for row in range(155,165):
        for col in range(99,110):
            weights[row,col,:]+=0.1

    for row in range(67,81):
        for col in range(163,168):
            weights[row,col,:]+=0.1
            
    for row in range(18,24):
        for col in range(86,94):
            weights[row,col,:]+=0.1
            
    for row in range(187,216):
        for col in range(249,252):
            weights[row,col,:]+=0.1
            
    for row in range(160,165):
        for col in range(157,162):
            weights[row,col,:]+=0.1
            
    for row in range(188,194):
        for col in range(90,95):
            weights[row,col,:]+=0.1
    
    for row in range(140,147):
        for col in range(158,160):
            weights[row,col,:]+=0.1
            
    for row in range(118,123):
        for col in range(134,141):
            weights[row,col,:]+=0.1
            
    for row in range(49,59):
        for col in range(123,128):
            weights[row,col,:]+=0.1
            
    for row in range(84,102):
        for col in range(87,106):
            weights[row,col,:]+=0.1
            
    for row in range(172,176):
        for col in range(187,192):
            weights[row,col,:]+=0.1
            
    for row in range(83,85):
        for col in range(96,101):
            weights[row,col,:]+=0.1
            
    for row in range(100,104):
        for col in range(95,100):
            weights[row,col,:]+=0.1

    for row in range(87,92):
        for col in range(85,90):
            weights[row,col,:]+=0.1
            
    for row in range(93,96):
        for col in range(105,108):
            weights[row,col,:]+=0.1
            
    for row in range(91,95):
        for col in range(94,101):
            weights[row,col,:]+=0.1
    
    for row in range(221,224):
        for col in range(213,217):
            weights[row,col,:]+=0.1
            
    for row in range(221,225):
        for col in range(235,238):
            weights[row,col,:]+=0.1

    with open("paris_weight_015.weight","w") as f:
        f.write("[")
        for row in range(256):
            for col in range(256):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(255,255,4):
                        f.write(",")
        f.write("]")    
        
gen_weights_paris_015()