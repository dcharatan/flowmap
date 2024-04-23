import torch
import torch.nn.functional as F


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]


def compute_out_of_boundary_mask(flow):
    # flow: [B, 2, H, W]
    assert flow.dim() == 4 and flow.size(1) == 2
    b, _, h, w = flow.shape
    init_coords = coords_grid(b, h, w).to(flow.device)
    corres = init_coords + flow  # [B, 2, H, W]

    max_w = w - 1
    max_h = h - 1

    valid_mask = (corres[:, 0] >= 0) & (corres[:, 0] <= max_w) & (corres[:, 1] >= 0) & (corres[:, 1] <= max_h)

    # in case very large flow
    flow_mask = (flow[:, 0].abs() <= max_w) & (flow[:, 1].abs() <= max_h)

    valid_mask = valid_mask & flow_mask

    return valid_mask  # [B, H, W]


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num
