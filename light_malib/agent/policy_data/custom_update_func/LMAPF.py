from light_malib.utils.logger import Logger

def update_func(policy_data_manager, eval_results, **kwargs):
    for policy_comb, agents_results in eval_results.items():
        agent_id, policy_id = policy_comb[0]
        results = agents_results[agent_id]
        idx = policy_data_manager.agents[agent_id].policy_id2idx[policy_id]
        
        for key in ["throughput"]:
            policy_data_manager.data[key][idx] = results[key]
            
    Logger.info(
        "update_func: {}".format(results)
    )