import torch
import torch.nn.functional as F

MOD_GET_FN = lambda self, key: self[int(key)] if isinstance(self, torch.nn.Sequential) \
                                              else getattr(self, key)


def load_batch_train(train_data, batch_idx, patch_size, affine_rand=0.05, fixed_patch_idx=None):
    # TODO refactor
    assert fixed_patch_idx in range(8) or fixed_patch_idx == None or fixed_patch_idx == 'center'

    num_batch = len(batch_idx)
    C = max(train_data.shape[1] - 1, 1)
    train_img = torch.zeros(num_batch, C, patch_size[0], patch_size[1], patch_size[2]).to(train_data.device)

    train_img1 = None
    train_label = torch.zeros(num_batch, patch_size[0], patch_size[1], patch_size[2]).to(train_data.device).long()

    for b in range(num_batch):
        with torch.no_grad():
            # Get patches
            data = train_data[batch_idx[b]]
            if fixed_patch_idx is None:
                rand_patch1 = torch.randint(max(data.shape[1] - patch_size[0], 0), (1,))
                rand_patch2 = torch.randint(max(data.shape[2] - patch_size[1], 0), (1,))
                rand_patch3 = torch.randint(max(data.shape[3] - patch_size[2], 0), (1,))
            elif fixed_patch_idx == 'center':
                rand_patch1 = max((data.shape[1]-patch_size[0])//2, 0)
                rand_patch2 = max((data.shape[2]-patch_size[1])//2, 0)
                rand_patch3 = max((data.shape[3]-patch_size[2])//2, 0)
            else:
                p_idxs = f"{fixed_patch_idx:03b}"
                p_idxs = [int(idx) for idx in [*p_idxs]]
                rand_patch1 = p_idxs[0] * patch_size[0]
                rand_patch2 = p_idxs[1] * patch_size[1]
                rand_patch3 = p_idxs[2] * patch_size[2]
                # print(rand_patch1, rand_patch2, rand_patch3)

            out_shape = (
                1,
                1,
                max(data.shape[1], patch_size[0]),
                max(data.shape[2], patch_size[1]),
                max(data.shape[3], patch_size[2]),
            )
            grid = F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(train_data.device)
                + affine_rand * torch.randn(1, 3, 4).to(train_data.device),
                out_shape, align_corners=False
            )
            patch_grid = grid[
                :,
                rand_patch1 : rand_patch1 + patch_size[0],
                rand_patch2 : rand_patch2 + patch_size[1],
                rand_patch3 : rand_patch3 + patch_size[2],
            ]
            if train_data.shape[1] > 1:
                train_label[b] = (
                    F.grid_sample(
                        data[-1:].unsqueeze(0).to(train_data.device), patch_grid, mode="nearest", align_corners=False
                    )
                    .squeeze()
                    .long()
                )
            train_img[b] = F.grid_sample(
                data[:-1].unsqueeze(0).to(train_data.device), patch_grid, align_corners=False
            ).squeeze()

    train_label = train_label.clamp_min_(0)

    return train_img, train_label


def soft_dice_loss(fixed,moving):
    # TODO refactor
    B,C,D,H,W = fixed.shape
    # TODO Add d parameter

    nominator = (4. * fixed*moving).reshape(B,-1,D*H*W).mean(2)
    denominator = ((fixed + moving)**2).reshape(B,-1,D*H*W).mean(2)

    if denominator.sum() == 0.:
        dice = (nominator * 0.) + 1.
    else:
        dice  = nominator / denominator # Do not add an eps here, it disturbs the consistency

    return dice



def fix_all(m):
    for p in m.parameters():
        p.requires_grad_(False)



def release_all(m):
    for p in m.parameters():
        p.requires_grad_(True)



def release_norms(m):
    if 'instancenorm' in m.__class__.__name__.lower() or 'batchnorm' in m.__class__.__name__.lower():
        print("Released", m.__class__.__name__)
        for p in m.parameters():
            p.requires_grad_(True)



import functools
def get_named_layers_leaves(module):
    """ Returns all leaf layers of a pytorch module and a keychain as identifier.
        e.g.
        ...
        ('features.0.5', nn.ReLU())
        ...
        ('classifier.0', nn.BatchNorm2D())
        ('classifier.1', nn.Linear())
    """

    return [(keychain, sub_mod) for keychain, sub_mod in list(module.named_modules()) if not next(sub_mod.children(), None)]



def get_module_data(module, keychain):
    """Retrieves any data inside a pytorch module for a given keychain.
       Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """

    return functools.reduce(MOD_GET_FN, keychain.split('.'), module)



def set_module_data(module, keychain, data):
    """Replaces any data inside a pytorch module for a given keychain.
       Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """
    key_list = keychain.split('.')
    root = functools.reduce(MOD_GET_FN, key_list[:-1], module)
    leaf = key_list[-1]
    if isinstance(root, torch.nn.Sequential):
        root[int(leaf)] = data
    else:
        setattr(root, leaf, data)