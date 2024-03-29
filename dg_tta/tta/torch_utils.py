from collections import OrderedDict

import torch
import torch.nn.functional as F

MOD_GET_FN = (
    lambda self, key: self[int(key)]
    if isinstance(self, torch.nn.Sequential)
    else getattr(self, key)
)


def get_batch(tensor_list, batch_idxs, patch_size, fixed_patch_idx=None, device="cpu"):
    assert (
        fixed_patch_idx in range(8)
        or fixed_patch_idx == None
        or fixed_patch_idx == "center"
    )

    device = torch.device(device)
    B = len(batch_idxs)
    b_img = []
    b_label = []

    t_patch_size = torch.as_tensor(patch_size)
    t_input_shape = torch.as_tensor(tensor_list[0].shape[-3:])
    scales = t_patch_size / t_input_shape
    scales = torch.cat([scales.flip(0), torch.tensor([1.])], dim=0)

    patch_affine = scales.diag()

    with torch.no_grad():
        for b in range(B):
            # Get patches
            data = tensor_list[batch_idxs[b]]

            if fixed_patch_idx == "center":
                pass
            else:
                rand_offset = 2.*torch.rand(3)-1.
                offset_range = (t_input_shape - t_patch_size) / t_input_shape
                offset_range = offset_range.clip(min=0.0)
                ranged_offset = rand_offset * offset_range
                ranged_offset = torch.cat([ranged_offset.flip(0), torch.tensor([1.])], dim=0)
                patch_affine[:,-1] = ranged_offset

            out_shape = (
                1,
                1,
                patch_size[0],
                patch_size[1],
                patch_size[2],
            )

            patch_grid = F.affine_grid(
                patch_affine[:3][None].to(device), out_shape, align_corners=False
            )
            img_min = data[0].min()
            img_patch = F.grid_sample(
                data[0][None,None].to(device) - img_min.to(device), patch_grid, align_corners=False, padding_mode="zeros"
            )
            img_patch = img_patch + img_min.to(device)
            # import nibabel as nib
            # nib.Nifti1Image(img_patch.squeeze().cpu().numpy(), torch.eye(4).numpy()).to_filename("out.nii.gz")
            b_img.append(img_patch)

            if data[1:].numel() == 0:
                # No GT label is available for this sample
                b_label.append(None)
            else:
                lbl_patch = F.grid_sample(
                    data[1:][None].to(device), patch_grid, align_corners=False, padding_mode="zeros", mode="nearest"
                )
                b_label.append(get_argmaxed_segs(lbl_patch))

    return b_img, b_label


def get_argmaxed_segs(segs):
    segs_oh_w_bg = torch.cat([(segs.sum(1, keepdim=True) < 1.0).float(), segs], dim=1)
    segs_argmaxed = segs_oh_w_bg.argmax(1, keepdim=True)
    return segs_argmaxed


def get_imgs(tta_sample):
    imgs = tta_sample[:, 0:1]
    return imgs


def soft_dice_loss(smp_a, smp_b):
    B, _, D, H, W = smp_a.shape
    d = 2

    nominator = (2.0 * smp_a * smp_b).reshape(B, -1, D * H * W).mean(2)
    denominator = 1 / d * ((smp_a + smp_b) ** d).reshape(B, -1, D * H * W).mean(2)

    if denominator.sum() == 0.0:
        dice = (nominator * 0.0) + 1.0
    else:
        dice = (
            nominator / denominator
        )  # Do not add an eps here, it disturbs the consistency

    return dice


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label - 1).fill_(0)

    for label_num in range(1, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2.0 * intersection) / (
            1e-8 + torch.mean(iflat) + torch.mean(tflat)
        )
    return dice


def fix_all(m):
    for p in m.parameters():
        p.requires_grad_(False)


def release_all(m):
    for p in m.parameters():
        p.requires_grad_(True)


def release_norms(m):
    if (
        "instancenorm" in m.__class__.__name__.lower()
        or "batchnorm" in m.__class__.__name__.lower()
    ):
        print("Released", m.__class__.__name__)
        for p in m.parameters():
            p.requires_grad_(True)


import functools


def get_named_layers_leaves(module):
    """Returns all leaf layers of a pytorch module and a keychain as identifier.
    e.g.
    ...
    ('features.0.5', nn.ReLU())
    ...
    ('classifier.0', nn.BatchNorm2D())
    ('classifier.1', nn.Linear())
    """

    return [
        (keychain, sub_mod)
        for keychain, sub_mod in list(module.named_modules())
        if not next(sub_mod.children(), None)
    ]


def get_module_data(module, keychain):
    """Retrieves any data inside a pytorch module for a given keychain.
    Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """

    return functools.reduce(MOD_GET_FN, keychain.split("."), module)


def set_module_data(module, keychain, data):
    """Replaces any data inside a pytorch module for a given keychain.
    Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """
    key_list = keychain.split(".")
    root = functools.reduce(MOD_GET_FN, key_list[:-1], module)
    leaf = key_list[-1]
    if isinstance(root, torch.nn.Sequential):
        root[int(leaf)] = data
    else:
        setattr(root, leaf, data)


def register_forward_pre_hook_at_beginning(model, hook_fn):
    hook_dict = {
        k: v
        for k, v in zip(
            range(len(model._forward_pre_hooks) + 1),
            [hook_fn] + list(model._forward_pre_hooks.values()),
        )
    }
    model._forward_pre_hooks = OrderedDict(hook_dict)


def register_forward_hook_at_beginning(model, hook_fn):
    hook_dict = {
        k: v
        for k, v in zip(
            range(len(model._forward_hooks) + 1),
            [hook_fn] + list(model._forward_hooks.values()),
        )
    }
    model._forward_hooks = OrderedDict(hook_dict)


def hookify(fn, type):
    assert type in ["forward_pre_hook", "forward_hook"]

    if type == "forward_pre_hook":
        return lambda module, input: fn(*input)
    elif type == "forward_hook":
        return lambda module, input, output: fn(output)

    raise ValueError()


def map_label(label, map_idxs, input_format):
    assert input_format in ["logits", "argmaxed"]

    if input_format == "logits":
        # We have a non argmaxed map, suppose that C dimension is label dimension
        mapped_label = label
        # Swap B,C and subselect
        mapped_label = mapped_label.transpose(0, 1)[map_idxs].transpose(0, 1)
    else:
        mapped_label = torch.zeros_like(label)
        for lbl_idx, map_idx in enumerate(map_idxs):
            mapped_label[label == map_idx] = lbl_idx

    return mapped_label


def generate_label_mapping(source_label_dict, target_label_dict):
    assert all([isinstance(k, str) for k in source_label_dict.keys()])
    assert all([isinstance(k, str) for k in target_label_dict.keys()])
    assert set(source_label_dict.keys()).intersection(
        target_label_dict.keys()
    ), "There are no intersecting label names in given dicts."
    mapped_label = []

    mapping_dict = dict.fromkeys(
        list(source_label_dict.keys()) + list(target_label_dict.keys())
    )

    for key in mapping_dict:
        if key in source_label_dict and key in target_label_dict:
            mapping_dict[key] = (source_label_dict[key], target_label_dict[key])

    return {k: v for k, v in mapping_dict.items() if v is not None}


def get_map_idxs(label_mapping: dict, optimized_labels: list, input_type):
    assert input_type in ["pretrain_labels", "tta_labels"]
    assert optimized_labels[0] == "background"

    # Generate idxs from label_mapping dict
    map_idxs_list = []
    for reduced_idx, eval_label in enumerate(optimized_labels):
        src_idx, target_idx = label_mapping[eval_label]
        # map_idxs_list = [tts_dict[k] for k,v in amos_bcv_dict.items()]
        append_idx = src_idx if input_type == "pretrain_labels" else target_idx
        map_idxs_list.append(append_idx)

    map_idxs = torch.as_tensor(map_idxs_list)

    return map_idxs
