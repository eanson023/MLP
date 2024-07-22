import numpy as np
import torch


def print_batch(batch):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            print("{}: size {}".format(k, str(batch[k].size())))
        elif isinstance(batch[k], list):
            print("{}: # item {}".format(k, len(batch[k])))
        else:
            print("{}: {}".format(k, batch[k]))


def istensor(data):
    return isinstance(data, torch.Tensor)


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def tensor2numpy(ptdata):
    return ptdata.detach().cpu().numpy()


def to_data(ptdata):
    if ptdata is None: return ptdata
    if isinstance(ptdata, list) or isinstance(ptdata, tuple):
        return [tensor2numpy(dt) for dt in ptdata]
    elif isinstance(ptdata, dict):
        return {k: tensor2numpy(dt) for k, dt in ptdata.items()}
    else:
        return tensor2numpy(ptdata)


def where(cond, x1, x2):
    """ Differentiable equivalent of np.where (or tf.where)
        Note that type of three variables should be same.
    Args:
        cond: condition
        x1: selected value if condition is 1 (True)
        x2: selected value if condition is 0 (False)
    """
    return (cond * x1) + ((1 - cond) * x2)


def loc2mask(loc, feat_mask):
    B, L = feat_mask.size()
    nfeatstamps = to_data(feat_mask.sum(dim=1))
    loc = to_data(loc)

    mask = np.zeros((B, L))
    for bi in range(B):
        sIdx = int(loc[bi, 0] * nfeatstamps[bi])
        eIdx = int(loc[bi, 1] * nfeatstamps[bi])
        mask[bi, sIdx:eIdx + 1] = 1

    return mask


""" Computation helpers """


def apply_on_sequence(layer, inp):
    " For nn.Linear, this fn is DEPRECATED "
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output


def time_to_index(start_time, end_time, num_units, duration):
    # Get the start time of the duration corresponding to each snippet code
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))  # 穷举
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap
