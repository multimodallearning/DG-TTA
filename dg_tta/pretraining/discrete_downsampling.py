
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from skimage.transform import resize



def augment_discrete_linear_downsampling_scipy(data_sample, zoom_range=(1/6, 1/4, 1/2),
    zoom_axes_invidually=False, p=.2,
    channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])

    if zoom_axes_invidually:
        zooms = np.random.choice(zoom_range, 3, replace=True)
    else:
        zooms = np.random.choice(zoom_range, 1)

    target_shape = np.round(shp * zooms).astype(int)

    if ignore_axes is not None:
        for i in ignore_axes:
            target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if np.random.uniform() < p:
            downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                 anti_aliasing=False)
            data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return data_sample



class SimulateDiscreteLowResolutionTransform(AbstractTransform):
    # Copy of SimulateLowResolutionTransform from batchgenerators

    def __init__(self, zoom_range=(1/6, 1/4, 1/2), zoom_axes_invidually=False,
                 per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.zoom_axes_invidually = zoom_axes_invidually
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_discrete_linear_downsampling_scipy(
                    data_dict[self.data_key][b],
                    zoom_range=self.zoom_range,
                    zoom_axes_invidually=self.zoom_axes_invidually,
                    p=self.p_per_channel,
                    channels=self.channels,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes)

        return data_dict