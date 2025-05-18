import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
import copy
from light_malib.utils.logger import Logger

def discrete_autoregreesive_act(decoder, obs_rep, obs, act_neighboring_masks, batch_size, n_agent, action_dim,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), device=obs.device, dtype=obs.dtype)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=obs.device)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32, device=obs.device)
    output_action_logits = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.long, device=obs.device)
    
    act_neighboring_masks = F.pad(act_neighboring_masks, (1,0), mode='constant', value=1)

    for i in range(n_agent):
        logit = decoder(shifted_action[:, i:i+1], obs_rep[:, i:i+1], obs[:, i:i+1], act_neighboring_masks[:, i:i+1, :i+1], start_pos=i)
        logit = logit.squeeze(1)
        
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        output_action_logits[:, i, :] = logit
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log, output_action_logits


def discrete_parallel_act(decoder, obs_rep, obs, act_neighboring_masks, action, batch_size, n_agent, action_dim,
                          available_actions=None):
    action=action.long()
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), device=obs.device, dtype=obs.dtype)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    
    act_neighboring_masks = F.pad(act_neighboring_masks, (1,0), mode='constant', value=1)[:,:,:-1]
    
    # start_pos = -1 means we don't use cache
    logit = decoder(shifted_action, obs_rep, obs, act_neighboring_masks, start_pos=-1)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy, logit


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim),requires_grad=False, device=obs.device, dtype=obs.dtype)
    output_action = torch.zeros((batch_size, n_agent, action_dim), device=obs.device, dtype=obs.dtype)
    output_action_log = torch.zeros_like(output_action, device=obs.device, dtype=obs.dtype)

    for i in range(n_agent):
        act_mean = decoder(shifted_action.clone(), obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim), device=obs.device, dtype=obs.dtype)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
