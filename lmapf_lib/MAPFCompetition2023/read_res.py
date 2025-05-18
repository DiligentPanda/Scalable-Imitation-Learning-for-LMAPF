folder="offline_eval/0112"
pattern_str="test_1t_095s_na(.*)"

import os
import json
import re

pattern=re.compile(pattern_str)

for fn in os.listdir(folder):
    if pattern.match(fn):
        fp=os.path.join(folder,fn)
        try:
            with open(fp) as f:
                data = json.load(f)
                throughput = data["numTaskFinished"]
                print(fn,throughput)
        except Exception as e:
            print(fp)
            # raise e
        