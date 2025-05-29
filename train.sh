# export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 # this would make everything very slow?
export RAY_DEDUP_LOGS=0

## Main Benchmark
### Imitiation Learning
# communicate with Spatially Sensitive Communication Module if "" 
# communicate with Attention Module if "_att"
# no communicate if _none
COMM_MODE="" 
python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_sortation_small_a600_s500${COMM_MODE}.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_warehouse_small_a600_s500${COMM_MODE}.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_paris_small_a600_s500${COMM_MODE}.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_berlin_small_a600_s500${COMM_MODE}.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_random_256_10_small_a600_s500${COMM_MODE}.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_random_256_20_small_a600_s500${COMM_MODE}.yaml

### Reinforcement Learning
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_sortation_small_a600_s500.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_warehouse_small_a600_s500.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_paris_small_a600_s500.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_berlin_small_a600_s500.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_random_256_10_small_a600_s500.yaml
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/rl_iter0_random_256_20_small_a600_s500.yaml

## Learn-to-follow Benchmark
### Imitiation Learning
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/ltf/Bootstrap/bootstrap_from_pibt_iter1_ltf_a256_r500.yaml
### Reinforcement Learning
# python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/ltf/RL/rl_iter0_ltf_a256_r500.yaml