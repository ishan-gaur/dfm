import torch
from typing import Callable, Optional, List
from dfm.generative_modeling import TransitionModel, TransitionFunc
from tqdm import tqdm
import random

SamplingStep = Callable[
    [TransitionFunc, torch.LongTensor, ...],
    [torch.FloatTensor, Optional[List[float]]],
]  # takes in transition kernel and current state and returns updated state. List of floats is the list of timepoints.

Integrator = Callable[
    [TransitionModel, List[str]], List[str]
]  # Takes in transition model, tokenizes the input and then runs sampling to completion; this will need other args for e.g. uniform sampling


def sample_any_order_ancestral(model: TransitionModel, x_SP: torch.LongTensor):
    mask_token_id = model.tokenizer.mask_token_id
    pad_token_id = model.tokenizer.pad_token_id

    # TODO[pi] generally we won't do this kind of coddling in our code, but this is the
    # main high level interface most users will interact with, so we're making an exception
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    x_SP = x_SP.to(model.device)

    t_st, t_end = 0.0, 1.0
    pbar = tqdm(total=t_end)

    t = t_st
    while t != t_end:
        x_SP = any_order_ancestral_step(model.transition_log_probs, x_SP, mask_token_id)
        len_S = x_SP.size(1) - (x_SP == pad_token_id).sum(dim=1)
        t_S = 1 - (x_SP == mask_token_id).sum(dim=1) / len_S
        t_new = t_S.min().item()
        pbar.update(t_new - t)
        # TODO[pi]: print out the matrix with the pbar updates so you can see it changing in place (this shouldn't print the matrix again an again causing vertical scrolling)
        t = t_new
    pbar.close()
    return x_SP.to(x_device)


def any_order_ancestral_step(
    transition_log_prob_fn: TransitionFunc,  # is there any reason not to just pass the model here?
    x_SP: torch.LongTensor,
    mask_token_id: int,
    next_pos_idx_SP: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    # TODO[pi] actually drop sequences that are finished and then place them in their original position at the end
    # so can we actually reduce the effective batch size if we can

    # If the caller doesn't specify which positions to sample next, sample a random masked index (if possible)
    # for each sequence
    if next_pos_idx_SP is None:
        next_pos_idx_SP = []  # idx tensor for the sequence and pos dimensions, doesn't actually have full SxP shape
        for s in range(x_SP.size(0)):
            masked_positions_S = (x_SP[s] == mask_token_id).nonzero().flatten()
            if len(masked_positions_S) > 0:
                rand_idx = random.randint(1, masked_positions_S.size(0)) - 1
                p = masked_positions_S[rand_idx]
                next_pos_idx_SP.append([s, p])
        next_pos_idx_SP = torch.LongTensor(next_pos_idx_SP)

    if next_pos_idx_SP.numel() == 0:
        return x_SP

    # Actually get the log probs and sample the change
    p_x_SPT = torch.exp(transition_log_prob_fn(x_SP))
    p_x_ST = torch.stack(
        [p_x_SPT[i, j, :] for i, j in next_pos_idx_SP]
    )  # note that some sequences (S dimension) might be missing
    x_new_S = torch.multinomial(p_x_ST, num_samples=1)
    for i_for_pos_idx, (i, j) in enumerate(next_pos_idx_SP):
        x_SP[i, j] = x_new_S[i_for_pos_idx]
    return x_SP
