# for warehouse_large.map

import numpy as np

def gen_weights_sortation_small_001():
    weights=np.ones((33,57,5),dtype=np.int32)*2
    
    c=0
    
    # : we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    # ##### boarder region #####
    # ## left 
    # for col in range(0,6):
    #     # see col 5 for consistency
    #     if col%2==1:
    #         # go south
    #         weights[:,col,1]=2-c
    #         # go north
    #         weights[:,col,3]=100000
    #     elif col%2==0:
    #         # go south
    #         weights[:,col,1]=100000
    #         # go north
    #         weights[:,col,3]=2-c      
    
    # ## top
    # for row in range(0,6):
    #     # see row 5 for consistency
    #     if row%2==1:
    #         # go east
    #         weights[row,:,0]=2-c
    #         # go west
    #         weights[row,:,2]=100000
    #     elif row%2==0:
    #         # go east
    #         weights[row,:,0]=100000
    #         # go west
    #         weights[row,:,2]=2-c
            
    # ## right
    # for col in range(493,500):
    #     # see col 493 for consistency
    #     if col%2==1:
    #         # go south
    #         weights[:,col,1]=2-c
    #         # go north
    #         weights[:,col,3]=100000
    #     elif col%2==0:
    #         # go south
    #         weights[:,col,1]=100000
    #         # go north
    #         weights[:,col,3]=2-c
        
    # ## bottom
    # for row in range(133,140):
    #     # see row 133 for consistency
    #     if row%2==1:
    #         # go east
    #         weights[row,:,0]=2-c
    #         # go west
    #         weights[row,:,2]=100000
    #     elif row%2==0:
    #         # go east
    #         weights[row,:,0]=100000
    #         # go west
    #         weights[row,:,2]=2-c
    
    ##### central region #####
    for row in range(5,28,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:52,0]=2-c
            # go west
            weights[row,5:52,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:52,0]=100000
            # go west
            weights[row,5:52,2]=2-c
        else:
            assert False,row
            
    for col in range(5,52,2):
        if (col-5)%4==0:
            # go south
            weights[5:28,col,1]=2-c
            # go north
            weights[5:28,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:28,col,1]=100000
            # go north
            weights[5:28,col,3]=2-c
        else:
            assert False
            
    with open("sortation_small_w001.w","w") as f:
        f.write("[")
        for row in range(33):
            for col in range(57):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(32,56,4):
                        f.write(",")
        f.write("]")

def gen_weights_sortation_large_001():
    weights=np.ones((140,500,5),dtype=np.int32)*2
    
    c=0
    
    # : we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    # ##### boarder region #####
    # ## left 
    # for col in range(0,6):
    #     # see col 5 for consistency
    #     if col%2==1:
    #         # go south
    #         weights[:,col,1]=2-c
    #         # go north
    #         weights[:,col,3]=100000
    #     elif col%2==0:
    #         # go south
    #         weights[:,col,1]=100000
    #         # go north
    #         weights[:,col,3]=2-c      
    
    # ## top
    # for row in range(0,6):
    #     # see row 5 for consistency
    #     if row%2==1:
    #         # go east
    #         weights[row,:,0]=2-c
    #         # go west
    #         weights[row,:,2]=100000
    #     elif row%2==0:
    #         # go east
    #         weights[row,:,0]=100000
    #         # go west
    #         weights[row,:,2]=2-c
            
    # ## right
    # for col in range(493,500):
    #     # see col 493 for consistency
    #     if col%2==1:
    #         # go south
    #         weights[:,col,1]=2-c
    #         # go north
    #         weights[:,col,3]=100000
    #     elif col%2==0:
    #         # go south
    #         weights[:,col,1]=100000
    #         # go north
    #         weights[:,col,3]=2-c
        
    # ## bottom
    # for row in range(133,140):
    #     # see row 133 for consistency
    #     if row%2==1:
    #         # go east
    #         weights[row,:,0]=2-c
    #         # go west
    #         weights[row,:,2]=100000
    #     elif row%2==0:
    #         # go east
    #         weights[row,:,0]=100000
    #         # go west
    #         weights[row,:,2]=2-c
    
    ##### central region #####
    for row in range(5,133,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:493,0]=2-c
            # go west
            weights[row,5:493,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:493,0]=100000
            # go west
            weights[row,5:493,2]=2-c
        else:
            assert False,row
            
    for col in range(5,493,2):
        if (col-5)%4==0:
            # go south
            weights[5:133,col,1]=2-c
            # go north
            weights[5:133,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:133,col,1]=100000
            # go north
            weights[5:133,col,3]=2-c
        else:
            assert False
            
    with open("sortation_large_w001.w","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")

gen_weights_sortation_small_001()
gen_weights_sortation_large_001()