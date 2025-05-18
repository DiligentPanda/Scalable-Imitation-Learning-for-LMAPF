import numpy as np
from scipy.stats import t as t_dist

def get_params(policy):
    
    pass

def set_params():
    pass




def get_stats(arr):
    sample_mean = np.mean(arr)
    sample_std = np.std(arr,ddof=1)
    sample_min = np.min(arr)
    sample_max = np.max(arr)
    sample_p025 = np.percentile(arr, 2.5)
    sample_p975 = np.percentile(arr, 97.5)
    n = len(arr)
    degree_of_freedom = n-1
    confidence_interval = t_dist.interval(
        0.95, 
        degree_of_freedom, 
        loc=sample_mean, 
        scale=sample_std / np.sqrt(n)
    )
    stats={
        "mean": sample_mean,
        "std": sample_std,
        "min": sample_min,
        "max": sample_max,
        "p025": sample_p025,
        "p975": sample_p975,
        "ci95": confidence_interval 
    }
    
    return stats