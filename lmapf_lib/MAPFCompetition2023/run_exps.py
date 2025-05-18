import subprocess
import multiprocessing

subprocess.check_output("./compile.sh", shell=True) 

instances = {
    # "warehouse_small_uniform": (500,600),
    # "sortation_small_uniform": (500,600),
    # "Paris_1_256_small": (500,600),
    # "Berlin_1_256_small": (500,600),
    # "random_256_10_small": (500,600),
    # "random_256_20_small": (500,600),
    # "warehouse_large_uniform": (3200,10000),
    # "sortation_large_uniform": (3200,10000),
    # "Paris_1_256": (2500,10000),
    # "Berlin_1_256": (2500,10000),
    # "random_256_10": (2500,10000),
    # "random_256_20": (2500,10000)
}



def run(cmd):
    try:
        print("start: ", cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCC] {}".format(cmd))
            # print(result.stdout)
        else:
            print("[FAIL] {}".format(cmd))
            # print(result.stdout)
            print(result.stderr)
    except Exception:
        import traceback
        print("[EXCEPTION]", traceback.format_exc())

max_process_num=4
num_runs=8

for instance, (num_steps,num_agents) in instances.items():
    pool=multiprocessing.Pool(max_process_num)
    cmds=[]
    for run_idx in range(num_runs):
        config_path=f"/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/configs/{instance}/{instance}_{run_idx}_{num_agents}.json"
        output_path=f"eval/{instance}/{run_idx}.json"
        cmd=f"LNS_NUM_THREADS=8 LNS_MAX_ITERATIONS=40000 ./build/lifelong --inputFile {config_path} -o {output_path} --simulationTime {num_steps} --planTimeLimit 30  --fileStoragePath large_files/"
        cmds.append(cmd) 
    pool.map(run, cmds)


        
        