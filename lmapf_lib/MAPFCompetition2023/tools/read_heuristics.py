import zlib
import os
import sys
import torch
import struct
 
def read(path):
    print("read from {}".format(path))
    
    if not os.path.isfile(path):
        raise Exception("File not found: {}".format(path))
     
    with open(path, 'rb') as f:
        compressed_data = f.read()

    # Decompress the data
    data = zlib.decompress(compressed_data)
    
    offset=0
    loc_size=struct.unpack('i', data[offset:offset+4])[0]
    offset+=4
    
    assert offset==4
    assert len(data)==4*(1+loc_size+loc_size*loc_size)
    
    empty_locs=torch.zeros(size=(loc_size,),dtype=torch.int32)
    for i in range(loc_size):
        empty_locs[i]=struct.unpack('i', data[offset:offset+4])[0]
        offset+=4
        
    assert offset==4*(1+loc_size)
    
    main_heuristics=torch.zeros(size=(loc_size,loc_size),dtype=torch.float64)
    for i in range(loc_size*loc_size):
        main_heuristics[i//loc_size,i%loc_size]=struct.unpack('f', data[offset:offset+4])[0]
        offset+=4

    # assert offset==len(data)

    print(empty_locs[:100], main_heuristics[0:2,:100])
    
    
import time    

s=time.time()
read("../../temp/test.zip")
elapse=time.time()-s
print(elapse)