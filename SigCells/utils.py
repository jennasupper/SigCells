import torch
from si4onnx.nn import truncated_interval


def threshold(
    thr, x, bias, a, b, l, u, apply_abs: bool = False, use_sigmoid: bool = False
):
    """
    Threshold the input tensor x with the given threshold (thr)
    and return the indices greater than thr.
    Args:
        thr (torch.Tensor): threshold tensor, dtype=torch.float64
        x (torch.Tensor): input tensor
        a (torch.Tensor): a tensor
        b (torch.Tensor): b tensor
        bias (torch.Tensor): bias tensor
        apply_abs (bool): apply abs to the input tensor or not
        use_sigmoid (bool): use sigmoid or not in the final output layer
    Returns:
        thresholed_index (torch.Tensor): thresholded index tensor
        l (float): lower bound of the truncated interval
        u (float): upper bound of the truncated interval
    """

    if use_sigmoid:
        tau = torch.logit(thr)
    else:
        tau = thr

    if apply_abs:
        abs_x = torch.abs(x)
        thresholed_index = abs_x > thr
        thresholed_index = torch.flatten(thresholed_index).int()

        negative_index = x < -thr
        tTa = torch.where(negative_index, a, -a)
        tTb = torch.where(negative_index, b, -b)
        event_bias = bias + tau
        event_bias = torch.where(negative_index, event_bias, -event_bias)
        l_negative, u_negative = truncated_interval(tTa, tTb, event_bias)
        l = torch.max(l, l_negative)
        u = torch.min(u, u_negative)
    else:
        thresholed_index = x > thr
        thresholed_index = torch.flatten(thresholed_index).int()

    positive_index = x > thr
    tTa = torch.where(positive_index, -a, a)
    tTb = torch.where(positive_index, -b, b)
    event_bias = bias - tau
    event_bias = torch.where(positive_index, -event_bias, event_bias)
    l_positive, u_positive = truncated_interval(tTa, tTb, event_bias)
    l = torch.max(l, l_positive)
    u = torch.min(u, u_positive)
    l = float(l.item())
    u = float(u.item())

    assert l < u
    return thresholed_index, l, u
