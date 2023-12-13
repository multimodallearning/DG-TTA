import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans



class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()

        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad = False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]

        if x_in.dim() == 5:
            mode = '3d'
        elif x_in.dim() == 4:
            mode = '2d'
        else:
            raise ValueError()

        if mode == '2d':
            nb, nc, nx, ny = x_in.shape
            ker = torch.randn([self.out_channel * nb, self.in_channel , k, k  ], requires_grad = self.requires_grad  ).cuda()
            shift = torch.randn( [self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0
            x_in = x_in.view(1, nb * nc, nx, ny)
            x_conv = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )

        elif mode == '3d':
            nb, nc, nx, ny, nz = x_in.shape
            ker = torch.randn([self.out_channel * nb, self.in_channel ,k ,k ,k], requires_grad = self.requires_grad  ).cuda()
            shift = torch.randn( [self.out_channel * nb, 1, 1 ,1], requires_grad = self.requires_grad  ).cuda() * 1.0
            x_in = x_in.view(1, nb * nc, nx, ny, nz)
            x_conv = F.conv3d(x_in, ker, stride =1, padding = k // 2, dilation = 1, groups = nb )
        else:
            raise ValueError()

        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        if mode == '2d':
            x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        elif mode == '3d':
            x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        else:
            raise ValueError()

        return x_conv



class GINGroupConv(nn.Module):
    def __init__(self, cfg):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = [1, 3 ] # don't make it tool large as we have multiple layers
        self.n_layer = cfg['N_LAYER']
        self.layers = []
        self.out_norm = 'frob'
        self.out_channel = cfg['IN_CHANNELS']
        in_channel = cfg['IN_CHANNELS']
        interm_channel = cfg['INTERM_CHANNELS']

        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = in_channel, scale_pool = self.scale_pool, layer_id = 0).cuda()
                )
        for ii in range(self.n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = interm_channel, scale_pool = self.scale_pool, layer_id = ii + 1).cuda()
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = self.out_channel, in_channel = interm_channel, scale_pool = self.scale_pool, layer_id = self.n_layer - 1, use_act = False).cuda()
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        if x_in.dim() == 5:
            mode = '3d'
        elif x_in.dim() == 4:
            mode = '2d'
        else:
            raise ValueError()

        if mode == '2d':
            nb, nc, nx, ny = x_in.shape
            alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1
        elif mode == '3d':
            nb, nc, nx, ny, nz = x_in.shape
            alphas = torch.rand(nb)[:, None, None, None, None] # nb, 1, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1, 1).cuda() # nb, nc, 1, 1, 1
        else:
            raise ValueError()

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            if mode == '2d':
                _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
                _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
                _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            elif mode == '3d':
                _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
                _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
                _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            else:
                raise ValueError()

            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed



def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()

    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)



def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img



def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist



class MIND3D():
    def __init__(self, delta=1, sigma=1) -> None:
        self.delta = delta
        self.sigma = sigma
        self.out_channels = 12
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.tensor([[0, 1, 1],
                                        [1, 1, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [2, 1, 1],
                                        [1, 2, 1]], dtype=torch.float)

        # squared distances
        dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :].long()
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :].long()
        mshift1 = torch.zeros((12, 1, 3, 3, 3))
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros((12, 1, 3, 3, 3))
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        self.rpad = nn.ReplicationPad3d(delta)
        self.mshift1 = mshift1
        self.mshift2 = mshift2

    def forward(self,img):
        # compute patch-ssd
        device = img.device
        ssd = smooth(((F.conv3d(self.rpad(img), self.mshift1.to(device), dilation=self.delta) - F.conv3d(self.rpad(img), self.mshift2.to(device), dilation=self.delta)) ** 2), self.sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind



class MIND2D_64(): #layout should be of size 2x64x1x2
    def __init__(self,image,layout,grid) -> None:
    #batch and channels should be equal to 1
        B,C,H,W = image.size()

        #smaller fixed length offsets for 64 MIND-SSC like features
        brief_layout3 = layout[0:1,0:,:,:]*0.15
        brief_layout4 = layout[1:2,0:,:,:]*0.15
        brief_layout4[:,:32,:,:] = 0
        fixed_length = 0.05
        brief_length = torch.sqrt(torch.sum((brief_layout3-brief_layout4)**2,3,keepdim=True))
        brief_layout3 /= (brief_length/fixed_length)
        brief_layout4 /= (brief_length/fixed_length)

        img_patch = F.unfold(image,5,padding=2).view(1,25,H,W)
        brief_patch = torch.sum((F.grid_sample(img_patch,brief_layout3+grid.view(1,1,-1,2),align_corners=True)-F.grid_sample(img_patch,brief_layout4+grid.view(1,1,-1,2),align_corners=True))**2,1)
        brief_patch -= brief_patch.min(1)[0]
        brief_patch /= torch.clamp_min(brief_patch.std(1),1e-5)
        brief_patch = torch.exp(-brief_patch).view(1,-1,grid.size(1),grid.size(2))

        return brief_patch



class nnUNetTrainer_GIN_MIND(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """MINDSSC nnUNet"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_input_channels = 12
        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        def gin_mind_hook(module, input):
            input = input[0]

            use_gin = '--tr_disable_internal_augmentation' not in sys.argv
            cfg = dict(
                IN_CHANNELS=1,
                N_LAYER=4,
                INTERM_CHANNELS=2,
            )
            gin_group_conv = GINGroupConv(cfg)
            input = gin_group_conv(input, use_gin)

            input = MIND3D().forward(input)
            return input

        network.register_forward_pre_hook(gin_mind_hook)

        return network