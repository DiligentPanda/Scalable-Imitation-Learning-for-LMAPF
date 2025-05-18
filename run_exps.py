import os

version="v3" # "v4"

exp_names=[
    # "sortation_small_uniform",
    # "sortation_small",
    # "paris_small",
    # "berlin_small",
    # "sortation_large_uniform",
    # "sortation_large",
    "paris_large",
    # "berlin_large"
    # "warehouse_small_uniform",
    # "warehouse_small",
    # "random_256_10_small",
    # "random_256_20_small",
    # "warehouse_large_uniform",
    # "warehouse_large",
    # "random_256_10_large",
    "random_256_20_large"
]

nums_agents=[
    # 600,
    # 6000,
    # 7000,
    8000,
    9000,
    10000,
    11000,
    12000
]

algorithms=[
    "PIBT",
    "PIBT-RL",
    "PIBT-IL"
    # "PIBT-LNS"
]

for num_agents in nums_agents:
    for algorithm in algorithms:
        for exp_name in exp_names:
            if algorithm in ["PIBT", "PIBT-RL"]:
                model_type="RL"
            else:
                model_type="IL"
            if algorithm in ["PIBT-RL", "PIBT-IL"]:
                WPPL_mode="PIBT-RL"
            elif algorithm in ["PIBT-LNS"]:
                WPPL_mode="PIBT-LNS"
            elif algorithm in ["PIBT"]:
                WPPL_mode="PIBT"
            else:
                raise ValueError("Invalid algorithm")
            model_name=exp_name.replace("_small", "").replace("_large", "").replace("_uniform", "")
            model_path="logs/main_reeval/{}/{}/{}/best".format(version, model_type, model_name)
            os.system("python evaluate.py {} {} {} {}".format(exp_name, model_path, WPPL_mode, num_agents))
        