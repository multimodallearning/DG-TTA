import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import numpy as np
import torch

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from dg_tta.tta.config_log_utils import (
    get_data_filepaths,
    get_dgtta_colormap,
    get_resources_dir,
)
from dg_tta.utils import check_dga_root_is_set


def read_image(source_data_paths, path_idx):
    if source_data_paths is None:
        return None, None

    source_img, source_sitk_stuff = SimpleITKIO().read_images(
        source_data_paths[path_idx : path_idx + 1]
    )
    source_img = source_img[0]
    return torch.tensor(source_img)[None, None, :], source_sitk_stuff


def get_target_imgs_datapaths():
    check_dga_root_is_set()
    with open("tta_plan.json", "r") as f:
        tta_plan = json.load(f)
    return  tta_plan["tta_data_filepaths"]


def get_source_imgs_datapaths():
    check_dga_root_is_set()
    buckets = ["imagesTr", "imagesTs"]
    with open("tta_plan.json", "r") as f:
        tta_plan = json.load(f)
    source_dataset_name = tta_plan["__pretrained_dataset_name__"]

    if source_dataset_name.startswith("TS104"):
        return "TS104"

    source_data_paths = []
    for buc in buckets:
        source_data_paths.extend(get_data_filepaths(source_dataset_name, buc))
    return source_data_paths


def get_orient_imgs(img):
    def get_axes_idxs(axis_size):
        NUM_IDXS = 16
        return np.linspace(0, axis_size - 1, NUM_IDXS).round().astype(int)

    img = img.squeeze(0, 1)
    D, H, W = img.shape
    slices = dict(HW=[], DW=[], DH=[])
    for d in get_axes_idxs(D):
        slices["HW"].append(img[d, :, :])
    for h in get_axes_idxs(H):
        slices["DW"].append(img[:, h, :])
    for w in get_axes_idxs(W):
        slices["DH"].append(img[:, :, w])
    return slices


def clear_axis(ax):
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def get_spacing_ratio(sitk_stuff, axis_idx):
    rolled_spacing = np.roll(np.array(sitk_stuff["spacing"]), axis_idx)
    return rolled_spacing[1] / rolled_spacing[0]


def show_image_overview(img, sitk_stuff, fig_inch_size=5.0):
    orient_imgs = get_orient_imgs(img)
    vmin, vmax = img.min(), img.max()

    dpi = 150.0
    large_text_size = fig_inch_size * 10
    small_text_size = fig_inch_size * 2
    cmap = get_dgtta_colormap()

    fig, fig_axes = plt.subplots(2, 2, figsize=(fig_inch_size, fig_inch_size), dpi=dpi)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    for fig_ax in fig_axes.flatten():
        clear_axis(fig_ax)

    for idx, (orientation, slices) in enumerate(orient_imgs.items()):
        current_fig_ax = fig_axes.flatten()[idx]
        grid_axes = ImageGrid(fig, 221 + idx, nrows_ncols=(4, 4), axes_pad=0.0)

        for ax, slc in zip(grid_axes, slices):
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_aspect(get_spacing_ratio(sitk_stuff, idx))

        for ax in grid_axes:
            clear_axis(ax)

        plt.figtext(
            0.5,
            0.5,
            orientation,
            ha="center",
            va="center",
            fontsize=large_text_size,
            color=cmap(idx / 3),
            transform=current_fig_ax.transAxes,
        )

    for fig_ax in fig_axes.flatten():
        fig_ax.set_facecolor("black")

    fig.set_facecolor("black")
    plt.figtext(
        0.5,
        0.5,
        f"Spacing: {[np.round(sp,1) for sp in sitk_stuff['spacing']]}",
        color="white",
        ha="center",
        va="center",
        fontsize=small_text_size,
        transform=fig_axes.flatten()[3].transAxes,
    )
    # plt.savefig("out.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close(fig)


def show_ts104_image():
    img_path = get_resources_dir() / "TS104_input_view.png"
    fig_inch_size = 7.0
    fig, ax = plt.subplots(dpi=150.0, figsize=(fig_inch_size, fig_inch_size))
    fig.set_facecolor("black")
    img = matplotlib.image.imread(img_path)
    ax.imshow(img)
    clear_axis(ax)
    ax.set_facecolor("black")
    plt.show()
    plt.close(fig)
