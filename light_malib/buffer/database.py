import h5py
import numpy as np
import torch
from collections import defaultdict

class Database:
    def __init__(self, path):
        self.path=path
        self.file=h5py.File(self.path,'r')
        
    def __del__(self):
        self.file.close()
        
    def load_data(self):
        meta=self.file["meta"]
        num_chunks=meta.attrs["num_chunks"]
        print(meta.attrs["num_episodes"],meta.attrs["mean_throughput"],meta.attrs["mean_step_time"], num_chunks)
        data=defaultdict(list)
        for chunk_idx in range(num_chunks):
            dataset_name="chunk_{}".format(chunk_idx)
            map_names=self.file["map_names/{}".format(dataset_name)][()]
            num_robots=self.file["num_robots/{}".format(dataset_name)][()]
            curr_positions=self.file["curr_positions/{}".format(dataset_name)][()]
            target_positions=self.file["target_positions/{}".format(dataset_name)][()]
            priorities=self.file["priorties/{}".format(dataset_name)][()]
            actions=self.file["actions/{}".format(dataset_name)][()]
            # print(curr_positions.shape, target_positions.shape, priorities.shape, actions.shape)
            
            # action_idxs, counts = np.unique(actions,return_counts=True)
            # print(action_idxs, counts)
            
            # B, L, A, ...
            for i in range(3):
                assert curr_positions.shape[i]==target_positions.shape[i]
                assert curr_positions.shape[i]==priorities.shape[i]
                assert curr_positions.shape[i]==actions.shape[i]
            
            #assert curr_positions.shape[0]==map_names.shape[0] and curr_positions.shape[0]==num_robots.shape[0], "{} vs {} vs {}".format(curr_positions.shape,map_names.shape,num_robots.shape)
            
            for idx in range(len(curr_positions)):
                _map_name=map_names[idx]#[idx+8*chunk_idx] #bug in some old datasets
                _map_name=_map_name.decode('ascii','ignore')
                _num_robots=num_robots[idx]#[idx+8*chunk_idx] #bug in some old datasets
                _curr_positions=curr_positions[idx]
                _target_positions=target_positions[idx]
                _priorities=priorities[idx]
                _actions=actions[idx]
                # _num_steps=_curr_positions.shape[0]
                
                # _curr_positions=_curr_positions.reshape(_num_steps,_num_robots)
                # _target_positions=target_positions.reshape(_num_steps,_num_robots)
                # _priorities=priorities.reshape(_num_steps,num_robots)
                # _actions=actions.reshape(_num_steps,num_robots)
                
                for jdx in range(_curr_positions.shape[0]):
                    datum={
                        "curr_positions":_curr_positions[jdx],
                        "target_positions":_target_positions[jdx],
                        "priorities":_priorities[jdx],
                        "actions":_actions[jdx]
                    }
                    data[(_map_name,_num_robots)].append(datum)
                    
                # datum={
                #     "map_name": _map_name,
                #     "num_robots": _num_robots,
                #     "curr_positions":_curr_positions[:200],
                #     "target_positions":_target_positions[:200],
                #     "priorities":_priorities[:200],
                #     "actions":_actions[:200]
                # }
                
        return data
    
    def udpate_no_instance_version(self, new_dataset_fp, _map_name, _num_robots):
        new_file=h5py.File(new_dataset_fp,'w')
        meta=self.file["meta"]
        num_chunks=meta.attrs["num_chunks"]
        
        print(meta.attrs["num_episodes"],meta.attrs["mean_throughput"],meta.attrs["mean_step_time"],num_chunks)
                
        new_file.create_group("meta")
        new_file["meta"].attrs["num_chunks"]=self.file["meta"].attrs["num_chunks"]
        new_file["meta"].attrs["num_episodes"]=self.file["meta"].attrs["num_episodes"]
        new_file["meta"].attrs["mean_throughput"]=self.file["meta"].attrs["mean_throughput"]
        new_file["meta"].attrs["mean_step_time"]=self.file["meta"].attrs["mean_step_time"]
        new_file.flush()
        
        for chunk_idx in range(num_chunks):
            if chunk_idx%32==0:
                print(f"Processing: {chunk_idx}/{num_chunks}")
            
            dataset_name="chunk_{}".format(chunk_idx)
            curr_positions=self.file["curr_positions/{}".format(dataset_name)][()]
            target_positions=self.file["target_positions/{}".format(dataset_name)][()]
            priorities=self.file["priorties/{}".format(dataset_name)][()]
            actions=self.file["actions/{}".format(dataset_name)][()]

            # sanity check
            # B, L, A, ...
            for i in range(3):
                assert curr_positions.shape[i]==target_positions.shape[i]
                assert curr_positions.shape[i]==priorities.shape[i]
                assert curr_positions.shape[i]==actions.shape[i]
            
            map_names=[_map_name.encode('ascii','ignore')]*curr_positions.shape[0]
            num_robots=[_num_robots]*curr_positions.shape[0]
            
            print(map_names,num_robots)
            
            new_file.create_dataset("map_names/{}".format(dataset_name),data=np.array(map_names))
            new_file.create_dataset("num_robots/{}".format(dataset_name),data=np.array(num_robots))
            new_file.create_dataset("curr_positions/{}".format(dataset_name), data=np.array(curr_positions))
            new_file.create_dataset("target_positions/{}".format(dataset_name), data=np.array(target_positions))
            new_file.create_dataset("priorties/{}".format(dataset_name), data=np.array(priorities))
            new_file.create_dataset("actions/{}".format(dataset_name), data=np.array(actions))
            
            
            new_file.flush()
            
        new_file.close()

            

if __name__=="__main__":
    database=Database("/root/GRF_MARL/datasets/paper_exps/sortation_small_a600_r512_e128_PIBT_RL_LNS_BC_i0.h5")
    #database.udpate_no_instance_version("/root/GRF_MARL/datasets/ltf_LNS_15_ITER_5000_EPISODES_PIBT_LNS_training-maps_v2.h5")
    data=database.load_data()
    s=0
    for k,v in data.items():
        print(k, len(v)/512)
        s+=len(v)/512
    print(len(data),s)