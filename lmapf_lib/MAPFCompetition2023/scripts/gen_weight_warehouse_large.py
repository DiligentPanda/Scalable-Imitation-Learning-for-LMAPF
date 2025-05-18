# for warehouse_large.map

import numpy as np

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
            
    with open("warehouse_large_weight_test3.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")
        
        
def gen_weights_warehouse_large_002():
    weights=np.ones((140,500,5),dtype=np.int32)*2

    c=0
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    ### central boarder ###
    
    for col in range(0,8,1):
        # see col 7 
        if (col-4)%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
        elif (col-4)%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
            
    for col in range(491,500,1):
        # see col 491
        if (col-491)%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
        elif (col-491)%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
    
    for row in range(0,8,1):
        # see row 7
        if (row-4)%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
        elif (row-4)%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
            
    for row in range(130,140,1):
        # see row 130
        if (row-130)%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
        elif (row-130)%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000        
    
    
    ### central ###
    
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
            
    with open("warehouse_large_weight_002.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")
        
def gen_weights_warehouse_large_003():
    weights=np.ones((140,500,5),dtype=np.int32)*2

    c=0
    
    #TODO(rivers): we need to add some consistent checking for weight assignment...
    #TODO(rivers): we need a tool to visualize weight map...
    
    # TODO(rivers): probably need better, make boarder and cental consistent
    ### boarder ###
    
    #### top ####
    for col in range(0,500,1): 
        # any
        if col%2==1:
            # go south
            weights[0:4,col,1]=2-c
            # go north
            weights[0:4,col,3]=100000
        elif col%2==0:
            # go south
            weights[0:4,col,1]=100000
            # go north
            weights[0:4,col,3]=2-c
            
    #### bottom ####
    for col in range(0,500,1): 
        # any
        if col%2==1:
            # go south
            weights[136:140,col,1]=2-c
            # go north
            weights[136:140,col,3]=100000
        elif col%2==0:
            # go south
            weights[136:140,col,1]=100000
            # go north
            weights[136:140,col,3]=2-c
    
    #### left ####
    for row in range(0,140,1):
        # any
        if row%2==1:
            # go east
            weights[row,0:4,0]=2-c
            # go west
            weights[row,0:4,2]=100000
        elif row%2==0:
            # go east
            weights[row,0:4,0]=100000
            # go west
            weights[row,0:4,2]=2-c
    
    
    #### right ####
    for row in range(0,140,1):
        # any
        if row%2==1:
            # go east
            weights[row,496:500,0]=2-c
            # go west
            weights[row,496:500,2]=100000
        elif row%2==0:
            # go east
            weights[row,496:500,0]=100000
            # go west
            weights[row,496:500,2]=2-c
    
    ### buffer ###
    
    for col in range(0,8,1):
        # see col 7 
        if (col-4)%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
        elif (col-4)%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
            
    for col in range(491,500,1):
        # see col 491
        if (col-491)%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=2-c
        elif (col-491)%2==1:
            # go south
            weights[:,col,1]=2-c
            # go north
            weights[:,col,3]=100000
    
    for row in range(0,8,1):
        # see row 7
        if (row-4)%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000
        elif (row-4)%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
            
    for row in range(130,140,1):
        # see row 130
        if (row-130)%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=2-c
        elif (row-130)%2==1:
            # go east
            weights[row,:,0]=2-c
            # go west
            weights[row,:,2]=100000        
    
    
    ### central ###
    
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
            
    with open("warehouse_large_weight_003.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")        
        
        

def gen_weights_warehouse_large_004():
    weights=np.ones((140,500,5),dtype=np.int32)*2
    
    large=100000 
    small=2

    up_weights=[
        small,large,small,large,large,small,large,small
    ]
    down_weights=[
        large,small,large,small,small,large,small,large
    ]
    
    for col in range(0,500):
        idx=col%8
        weights[:,col,1]=down_weights[idx]
        weights[:,col,3]=up_weights[idx]
    
    for row in range(0,140):
        if row%2==0:
            weights[row,:,0]=large
            weights[row,:,2]=small
        elif row%2==1:
            weights[row,:,0]=small+inc
            weights[row,:,2]=large

    with open("warehouse_large_weight_004.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")            

def gen_weights_warehouse_large_005():
    weights=np.ones((140,500,5),dtype=np.float32)
    
    for large in [1.2,1.5,2,2.5,3,4,5,10,20,50]:

        small=1

        up_weights=[
            small,large,small,large,large,small,large,small
        ]
        down_weights=[
            large,small,large,small,small,large,small,large
        ]
        
        for col in range(0,500):
            idx=col%8
            weights[:,col,1]=down_weights[idx]
            weights[:,col,3]=up_weights[idx]
        
        for row in range(0,140):
            if row%2==0:
                weights[row,:,0]=large
                weights[row,:,2]=small
            elif row%2==1:
                weights[row,:,0]=small
                weights[row,:,2]=large

        with open("warehouse_large_weight_value_{}.weight".format(large),"w") as f:
            f.write("[")
            for row in range(140):
                for col in range(500):
                    for dir in range(5):
                        f.write(str(weights[row,col,dir]))
                        if (row,col,dir)!=(139,499,4):
                            f.write(",")
            f.write("]")   


def gen_weights_warehouse_large_007():
    weights=np.ones((140,500,5),dtype=float)
    
    large=1000000
    small=1

    up_weights=[
        small,large,small,large,large,small,large,small
    ]
    down_weights=[
        large,small,large,small,small,large,small,large
    ]
    
    for col in range(0,500):
        idx=col%8
        weights[:,col,1]=down_weights[idx]
        weights[:,col,3]=up_weights[idx]
    
    for row in range(0,140):
        if row>=3 and row<=7:
            inc=0.01
        elif row>=130 and row<=136:
            inc=0.01
        else:
            inc=0
        if row%2==0:
            weights[row,:,0]=large
            weights[row,:,2]=small+inc
        elif row%2==1:
            weights[row,:,0]=small+inc
            weights[row,:,2]=large

    with open("warehouse_large_weight_007.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")            

def gen_weights_warehouse_large_008():
    weights=np.ones((140,500,5),dtype=float)
    
    large=1000000
    small=1

    up_weights=[
        small,large,small,large,large,small,large,small
    ]
    down_weights=[
        large,small,large,small,small,large,small,large
    ]
    
    for col in range(0,500):
        if col>=50 and col<450:
            inc=0.01
        else:
            inc=0
        idx=col%8
        weights[:,col,1]=down_weights[idx]+inc if down_weights[idx]==small else down_weights[idx]
        weights[:,col,3]=up_weights[idx]+inc if up_weights[idx]==small else up_weights[idx]
    
    for row in range(0,140):
        if row>=3 and row<=7:
            inc=0.01
        elif row>=130 and row<=136:
            inc=0.01
        else:
            inc=0
        if row%2==0:
            weights[row,:,0]=large
            weights[row,:,2]=small+inc
        elif row%2==1:
            weights[row,:,0]=small+inc
            weights[row,:,2]=large

    with open("warehouse_large_weight_008.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")       



gen_weights_warehouse_large_008()