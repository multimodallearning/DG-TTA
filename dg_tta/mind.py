import torch
import torch.nn.functional as F


def filter1D(img, weight, dim, padding_mode="replicate"):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(
        6,
    )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(
        5,
    )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(
        F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
        weight.view(view),
    ).view(B, C, D, H, W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(
        -torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2)
        / (2 * torch.pow(sigma, 2))
    )
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


class MIND2D_64:  # layout should be of size 2x64x1x2
    def __init__(self, image, layout, grid) -> None:
        # batch and channels should be equal to 1
        B, C, H, W = image.size()

        # smaller fixed length offsets for 64 MIND-SSC like features
        brief_layout3 = layout[0:1, 0:, :, :] * 0.15
        brief_layout4 = layout[1:2, 0:, :, :] * 0.15
        brief_layout4[:, :32, :, :] = 0
        fixed_length = 0.05
        brief_length = torch.sqrt(
            torch.sum((brief_layout3 - brief_layout4) ** 2, 3, keepdim=True)
        )
        brief_layout3 /= brief_length / fixed_length
        brief_layout4 /= brief_length / fixed_length

        img_patch = F.unfold(image, 5, padding=2).view(1, 25, H, W)
        brief_patch = torch.sum(
            (
                F.grid_sample(
                    img_patch,
                    brief_layout3 + grid.view(1, 1, -1, 2),
                    align_corners=True,
                )
                - F.grid_sample(
                    img_patch,
                    brief_layout4 + grid.view(1, 1, -1, 2),
                    align_corners=True,
                )
            )
            ** 2,
            1,
        )
        brief_patch -= brief_patch.min(1)[0]
        brief_patch /= torch.clamp_min(brief_patch.std(1), 1e-5)
        brief_patch = torch.exp(-brief_patch).view(1, -1, grid.size(1), grid.size(2))

        return brief_patch


class MIND3D(torch.nn.Module):
    def __init__(self, delta=1, sigma=1, randn_weighting=0.05) -> None:
        super().__init__()
        self.delta = delta
        self.sigma = sigma
        self.out_channels = 12
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.tensor(
            [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
            dtype=torch.float,
        )

        # squared distances
        dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        mask = (x > y).view(-1) & (dist == 2).view(-1)

        # build kernel
        idx_shift1 = (
            six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
        )
        idx_shift2 = (
            six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
        )
        mshift1 = torch.zeros((12, 1, 3, 3, 3))
        mshift1.view(-1)[
            torch.arange(12) * 27
            + idx_shift1[:, 0] * 9
            + idx_shift1[:, 1] * 3
            + idx_shift1[:, 2]
        ] = 1
        mshift2 = torch.zeros((12, 1, 3, 3, 3))
        mshift2.view(-1)[
            torch.arange(12) * 27
            + idx_shift2[:, 0] * 9
            + idx_shift2[:, 1] * 3
            + idx_shift2[:, 2]
        ] = 1
        self.rpad = torch.nn.ReplicationPad3d(delta)
        self.mshift1 = mshift1
        self.mshift2 = mshift2
        self.randn_weighting = randn_weighting

    def forward(self, img):
        # compute patch-ssd
        device = img.device

        edge_selection = F.conv3d(
            self.rpad(img), self.mshift1.to(device), dilation=self.delta
        ) - F.conv3d(self.rpad(img), self.mshift2.to(device), dilation=self.delta)

        edge_selection = edge_selection + self.randn_weighting * torch.randn_like(
            edge_selection
        )
        ssd = smooth(edge_selection**2, self.sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(
            mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000
        )
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind


def mind_hook(module, input):
    return MIND3D().forward(*input)
