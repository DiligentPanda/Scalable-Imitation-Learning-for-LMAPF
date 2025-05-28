# export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 # this would make everything very slow?
export RAY_DEDUP_LOGS=0
python light_malib/main_ppo.py --config expr_configs/paper_exps_v3/small/bootstrap_from_pibt_iter1_warehouse_small_a600_s500.yaml