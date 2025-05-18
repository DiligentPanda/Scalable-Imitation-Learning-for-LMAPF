# for warehouse_large.map

import numpy as np

def gen_weights_sortation_large_003():
    weights=np.ones((140,500,5),dtype=np.int32)*2
    
    c=0
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
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
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
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
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
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
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
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
            
    with open("sortation_large_weight_003.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")
        
        
def gen_weights_sortation_large_004():
    weights=np.ones((140,500,5),dtype=np.int32)*2
    
    c=0
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
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
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,0:6,0]=2-c
            # go west
            weights[row,0:6,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,0:6,0]=100000
            # go west
            weights[row,0:6,2]=2-c         
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
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
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[0:6,col,1]=2-c
            # go north
            weights[0:6,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[0:6,col,1]=100000
            # go north
            weights[0:6,col,3]=2-c           
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
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
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,493:500,0]=2-c
            # go west
            weights[row,493:500,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,493:500,0]=100000
            # go west
            weights[row,493:500,2]=2-c       
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
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
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[133:140,col,1]=2-c
            # go north
            weights[133:140,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[133:140,col,1]=100000
            # go north
            weights[133:140,col,3]=2-c       
    
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
            
    with open("sortation_large_weight_004.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")

def gen_weights_sortation_large_005():
    weights=np.ones((140,500,5),dtype=float)
    
    inc=0.01
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,0:6,0]=1
            # go west
            weights[row,0:6,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,0:6,0]=100000
            # go west
            weights[row,0:6,2]=1         
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=1+inc
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=1+inc
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[0:6,col,1]=1
            # go north
            weights[0:6,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[0:6,col,1]=100000
            # go north
            weights[0:6,col,3]=1           
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,493:500,0]=1
            # go west
            weights[row,493:500,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,493:500,0]=100000
            # go west
            weights[row,493:500,2]=1       
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=1+inc
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=1+inc
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[133:140,col,1]=1
            # go north
            weights[133:140,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[133:140,col,1]=100000
            # go north
            weights[133:140,col,3]=1       
    
    ##### central region #####
    for row in range(5,133,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:493,0]=1
            # go west
            weights[row,5:493,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:493,0]=100000
            # go west
            weights[row,5:493,2]=1
        else:
            assert False,row
            
    for col in range(5,493,2):
        if (col-5)%4==0:
            # go south
            weights[5:133,col,1]=1
            # go north
            weights[5:133,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:133,col,1]=100000
            # go north
            weights[5:133,col,3]=1
        else:
            assert False
            
    with open("sortation_large_weight_005.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write(str(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")




def gen_weights_sortation_large_006():
    weights=np.ones((140,500,5),dtype=float)
    
    inc=0.01
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,0:6,0]=1
            # go west
            weights[row,0:6,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,0:6,0]=100000
            # go west
            weights[row,0:6,2]=1         
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=1+inc-0.001*abs(row)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=1+inc-0.001*abs(row)
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[0:6,col,1]=1
            # go north
            weights[0:6,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[0:6,col,1]=100000
            # go north
            weights[0:6,col,3]=1           
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,493:500,0]=1
            # go west
            weights[row,493:500,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,493:500,0]=100000
            # go west
            weights[row,493:500,2]=1       
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=1+inc-0.001*abs(row-139)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=1+inc-0.001*abs(row-139)
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[133:140,col,1]=1
            # go north
            weights[133:140,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[133:140,col,1]=100000
            # go north
            weights[133:140,col,3]=1       
    
    ##### central region #####
    for row in range(5,133,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:493,0]=1
            # go west
            weights[row,5:493,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:493,0]=100000
            # go west
            weights[row,5:493,2]=1
        else:
            assert False,row
            
    for col in range(5,493,2):
        if (col-5)%4==0:
            # go south
            weights[5:133,col,1]=1
            # go north
            weights[5:133,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:133,col,1]=100000
            # go north
            weights[5:133,col,3]=1
        else:
            assert False
            
    with open("sortation_large_weight_006.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write("{:.3f}".format(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")

def gen_weights_sortation_large_007():
    weights=np.ones((140,500,5),dtype=float)
    
    inc=0.01
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1+(0.01 if col==0 else 0)  
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1+(0.01 if col==0 else 0)     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,0:6,0]=1
            # go west
            weights[row,0:6,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,0:6,0]=100000
            # go west
            weights[row,0:6,2]=1         
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=max(1+inc-0.001*abs(row)+0.007,1.0)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=max(1+inc-0.001*abs(row)+0.007,1.0)
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[0:6,col,1]=1
            # go north
            weights[0:6,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[0:6,col,1]=100000
            # go north
            weights[0:6,col,3]=1           
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1+(0.01 if col==499 else 0)     
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1+(0.01 if col==499 else 0)     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,493:500,0]=1
            # go west
            weights[row,493:500,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,493:500,0]=100000
            # go west
            weights[row,493:500,2]=1       
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=max(1+inc-0.001*abs(row-139)+0.007,1.0)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=max(1+inc-0.001*abs(row-139)+0.007,1.0)
            
    for col in range(0,500):
        if col%4==0 or col%4==1:
            # go south
            weights[133:140,col,1]=1
            # go north
            weights[133:140,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[133:140,col,1]=100000
            # go north
            weights[133:140,col,3]=1       
    
    ##### central region #####
    for row in range(5,133,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:493,0]=1
            # go west
            weights[row,5:493,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:493,0]=100000
            # go west
            weights[row,5:493,2]=1
        else:
            assert False,row
            
    for col in range(5,493,2):
        if (col-5)%4==0:
            # go south
            weights[5:133,col,1]=1
            # go north
            weights[5:133,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:133,col,1]=100000
            # go north
            weights[5:133,col,3]=1
        else:
            assert False
            
    with open("sortation_large_weight_007.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write("{:.3f}".format(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")



def gen_weights_sortation_large_008():
    weights=np.ones((140,500,5),dtype=float)
    
    inc=0.01
    
    # TODO(rivers): we need to test if row 0,139 and col 0,499 should be restrict.
    # that is we don't want the robot to go across potential task points, which may lead to more congestion.
    
    ##### boarder region #####
    ## left 
    for col in range(0,6):
        # see col 5 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1+(0.01 if col==0 else 0)  
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1+(0.01 if col==0 else 0)     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,0:6,0]=1
            # go west
            weights[row,0:6,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,0:6,0]=100000
            # go west
            weights[row,0:6,2]=1         
    
    ## top
    for row in range(0,6):
        # see row 5 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=max(1+inc-0.001*abs(row)+0.01,1.0)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=max(1+inc-0.001*abs(row)+0.01,1.0)
            
    for col in range(0,500):
        if col>=50 and col<450:
            inc=0.01
        else:
            inc=0
        if col%4==0 or col%4==1:
            # go south
            weights[0:6,col,1]=1+inc
            # go north
            weights[0:6,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[0:6,col,1]=100000
            # go north
            weights[0:6,col,3]=1+inc           
            
    ## right
    for col in range(493,500):
        # see col 493 for consistency
        if col%2==1:
            # go south
            weights[:,col,1]=1+(0.01 if col==499 else 0)     
            # go north
            weights[:,col,3]=100000
        elif col%2==0:
            # go south
            weights[:,col,1]=100000
            # go north
            weights[:,col,3]=1+(0.01 if col==499 else 0)     
            
    for row in range(0,140):
        if row%4==0 or row%4==1:
            # go east
            weights[row,493:500,0]=1
            # go west
            weights[row,493:500,2]=100000
        elif row%4==2 or row%4==3:
            # go east
            weights[row,493:500,0]=100000
            # go west
            weights[row,493:500,2]=1       
        
    ## bottom
    for row in range(133,140):
        # see row 133 for consistency
        if row%2==1:
            # go east
            weights[row,:,0]=max(1+inc-0.001*abs(row-139)+0.007,1.0)
            # go west
            weights[row,:,2]=100000
        elif row%2==0:
            # go east
            weights[row,:,0]=100000
            # go west
            weights[row,:,2]=max(1+inc-0.001*abs(row-139)+0.007,1.0)
            
    for col in range(0,500):
        if col>=50 and col<450:
            inc=0.01
        else:
            inc=0
        if col%4==0 or col%4==1:
            # go south
            weights[133:140,col,1]=1+inc
            # go north
            weights[133:140,col,3]=100000
        elif col%4==2 or col%4==3:
            # go south
            weights[133:140,col,1]=100000
            # go north
            weights[133:140,col,3]=1+inc
    
    ##### central region #####
    for row in range(5,133,2):
        if (row-5)%4==0:
            # go east
            weights[row,5:493,0]=1
            # go west
            weights[row,5:493,2]=100000
        elif (row-5)%4==2:
            # go east
            weights[row,5:493,0]=100000
            # go west
            weights[row,5:493,2]=1
        else:
            assert False,row
            
    for col in range(5,493,2):
        if col>=50 and col<450:
            inc=0.01
        else:
            inc=0
        if (col-5)%4==0:
            # go south
            weights[5:133,col,1]=1+inc
            # go north
            weights[5:133,col,3]=100000
        elif (col-5)%4==2:
            # go south
            weights[5:133,col,1]=100000
            # go north
            weights[5:133,col,3]=1+inc
        else:
            assert False
            
    
            
    with open("sortation_large_weight_008.weight","w") as f:
        f.write("[")
        for row in range(140):
            for col in range(500):
                for dir in range(5):
                    f.write("{:.3f}".format(weights[row,col,dir]))
                    if (row,col,dir)!=(139,499,4):
                        f.write(",")
        f.write("]")





gen_weights_sortation_large_008()