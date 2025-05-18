import glob
import re
import os
import pprint
import subprocess

glob_pattern_str="our_problems/**/*.json"
re_pattern_str="random-32-32-20_20_(.*).json"
args="-o test.json --simulationTime 5000 --planTimeLimit 1 --fileStoragePath large_files/"

instance_fps=glob.glob(glob_pattern_str,recursive=True)
pattern=re.compile(re_pattern_str)
instance_fps=[instance_fp for instance_fp in instance_fps if pattern.match(os.path.split(instance_fp)[-1])]

pprint.pprint(instance_fps)

if not os.path.exists("large_files"):
    os.makedirs("large_files",exist_ok=True)

print(os.listdir())
subprocess.check_call("cd build && make -j && cd ..",shell=True)

for instance_fp in instance_fps:
    print("[START] run {}".format(instance_fp))
    subprocess.check_call(
        "{executable} --inputFile {instance_fp} {args}".format(
            executable="./build/lifelong",
            instance_fp=instance_fp,
            args=args      
        ),
        shell=True
    )
    print("[ END ] run {}".format(instance_fp))
    