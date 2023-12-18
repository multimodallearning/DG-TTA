import torch
import torch.nn.functional as F

from dg_tta.mind import MIND3D
from dg_tta.gin import gin_aug


def get_rf_field(
    num_batch, size_3d, interpolation_factor=4, num_fields=4, device="cpu"
):
    rf_field = F.interpolate(
        F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(
                    torch.randn(
                        num_batch,
                        num_fields,
                        size_3d[0] // interpolation_factor,
                        size_3d[1] // interpolation_factor,
                        size_3d[2] // interpolation_factor,
                        device=device,
                    ),
                    interpolation_factor,
                    stride=1,
                    padding=interpolation_factor // 2,
                ),
                interpolation_factor,
                stride=1,
                padding=interpolation_factor // 2,
            ),
            interpolation_factor,
            stride=1,
            padding=interpolation_factor // 2,
        ),
        size=size_3d,
        mode="trilinear",
    )
    rf_field -= rf_field.mean((-3, -2, -1), keepdim=True)
    rf_field /= 1e-3 + rf_field.view(num_batch * num_fields, -1).std(1).view(
        num_batch, num_fields, 1, 1, 1
    )

    return rf_field


def calc_consistent_diffeomorphic_field(
    disp_field,
    inverse_disp_field,
    time_steps=1,
    ensure_inverse_consistency=True,
    iter_steps_override=None,
):
    # https://github.com/multimodallearning/convexAdam/blob/76a595914eb21ea17795e6cd19503ab447f0ea6b/l2r_2021_convexAdam_task1_docker.py#L166
    # https://github.com/cwmok/LapIRN/blob/d8f96770a704b1f190955cc26297c7b01a270b0a/Code/miccai2020_model_stage.py#L761

    # Vincent Arsigny, Olivier Commowick, Xavier Pennec, Nicholas Ayache: A Log-Euclidean Framework for Statistics on Diffeomorphisms
    B, C, D, H, W = disp_field.size()
    dimension_correction = torch.tensor([D, H, W], device=disp_field.device).view(
        1, 3, 1, 1, 1
    )
    dt = 1.0 / time_steps

    with torch.no_grad():
        identity = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0), (1, 1, D, H, W), align_corners=True
            )
            .permute(0, 4, 1, 2, 3)
            .to(disp_field)
        )
        if ensure_inverse_consistency:
            out_disp_field = (
                disp_field / dimension_correction / (2**time_steps) * dt
            ).clone()
            out_inverse_disp_field = (
                inverse_disp_field / dimension_correction / (2**time_steps) * dt
            ).clone()

            for _ in range(
                time_steps if not iter_steps_override else iter_steps_override
            ):
                ds = out_disp_field.clone()
                inverse_ds = out_inverse_disp_field.clone()
                out_disp_field = +0.5 * ds - 0.5 * F.grid_sample(
                    inverse_ds,
                    (identity + ds).permute(0, 2, 3, 4, 1),
                    padding_mode="border",
                    align_corners=True,
                )

                out_inverse_disp_field = +0.5 * inverse_ds - 0.5 * F.grid_sample(
                    ds,
                    (identity + inverse_ds).permute(0, 2, 3, 4, 1),
                    padding_mode="border",
                    align_corners=True,
                )
            out_disp_field = out_disp_field * 2**time_steps * dimension_correction
            out_inverse_disp_field = (
                out_inverse_disp_field * 2**time_steps * dimension_correction
            )

        else:
            # https://github.com/cwmok/LapIRN/blob/d8f96770a704b1f190955cc26297c7b01a270b0a/Code/miccai2020_model_stage.py#L761

            ds_dt = (
                disp_field / dimension_correction / (2**time_steps)
            )  # velocity = ds/dt
            inverse_ds_dt = (
                inverse_disp_field / dimension_correction / (2**time_steps)
            )
            ds = ds_dt * dt
            inverse_ds = inverse_ds_dt * dt

            for _ in range(
                time_steps if not iter_steps_override else iter_steps_override
            ):
                ds = ds + F.grid_sample(
                    ds,
                    (identity + ds).permute(0, 2, 3, 4, 1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                inverse_ds = inverse_ds + F.grid_sample(
                    inverse_ds,
                    (identity + inverse_ds).permute(0, 2, 3, 4, 1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )

            out_disp_field = ds * dimension_correction
            out_inverse_disp_field = inverse_ds * dimension_correction

    return out_disp_field, out_inverse_disp_field


def get_disp_field(
    batch_num, size_3d, factor=0.1, interpolation_factor=5, device="cpu"
):
    field = get_rf_field(
        batch_num,
        size_3d,
        alternating_fields=False,
        num_fields=3,
        interpolation_factor=interpolation_factor,
        device=device,
    )
    STEPS = 5
    disp_field, inverse_disp_field = calc_consistent_diffeomorphic_field(
        field * factor, torch.zeros_like(field), STEPS, ensure_inverse_consistency=True
    )
    return disp_field.permute(0, 2, 3, 4, 1), inverse_disp_field.permute(0, 2, 3, 4, 1)


def get_rand_affine(batch_size, strength=0.05, flip=False):
    affine = torch.cat(
        (
            torch.randn(batch_size, 3, 4) * strength + torch.eye(3, 4).unsqueeze(0),
            torch.tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(batch_size, 1, 1),
        ),
        1,
    )

    if flip:
        flip_affine = torch.diag(
            torch.cat([(2 * (torch.rand(3) > 0.5).float() - 1), torch.tensor([1.0])])
        )
        affine = affine @ flip_affine
    return affine[:, :3], affine.inverse()[:, :3]


def gin_mind_aug(input):
    return MIND3D()(gin_aug(input))
