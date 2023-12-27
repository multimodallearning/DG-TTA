import os
from itertools import chain
from typing import List, Union
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from dg_tta.tta.config_log_utils import suppress_stdout


def get_data_iterator(
    config, predictor, tta_data_filepaths, dataset_raw_path, tta_dataset_bucket
):
    assert tta_dataset_bucket in ["imagesTs", "imagesTr"]

    list_of_lists = [
        [_path]
        for _path in tta_data_filepaths
        if Path(_path).parts[-2] == tta_dataset_bucket
    ]

    label_folder = "labelsTs" if tta_dataset_bucket == "imagesTs" else "labelsTr"
    output_folder = (
        "tta_outputTs" if tta_dataset_bucket == "imagesTs" else "tta_outputTr"
    )

    (
        list_of_lists_or_source_folder,
        output_filename_truncated,
        seg_from_prev_stage_files,
    ) = predictor._manage_input_and_output_lists(
        list_of_lists, output_folder, dataset_raw_path / label_folder
    )

    if len(list_of_lists_or_source_folder) == 0:
        return iter(()), 0

    seg_from_prev_stage_files = [
        s if Path(s).is_file() else None for s in seg_from_prev_stage_files
    ]
    data_iterator = get_data_iterator_from_lists_of_filenames(
        predictor,
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
    )
    return data_iterator, len(list_of_lists)


def load_tta_data(config, dataset_raw_path, predictor, tta_across_all_samples=False):
    with suppress_stdout():
        ts_iterator, ts_data_len = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTs",
        )
        tr_iterator, tr_data_len = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTr",
        )

    if tta_across_all_samples:
        data = list(ts_iterator) + list(tr_iterator), ts_data_len + tr_data_len
    else:
        data = chain(ts_iterator, tr_iterator), ts_data_len + tr_data_len

    return data


def load_network(weights_file, device):
    pretrained_weights_filepath = Path(*Path(weights_file).parts[:-2])
    fold = Path(weights_file).parts[-2].replace("fold_", "")
    use_folds = [int(fold)] if fold.isnumeric() else fold
    checkpoint_name = Path(weights_file).parts[-1]
    configuration = Path(weights_file).parts[-3].split("__")[-1]

    perform_everything_on_gpu = True
    verbose = False

    predictor = nnUNetPredictor(
        perform_everything_on_gpu=perform_everything_on_gpu,
        device=device,
        verbose_preprocessing=verbose,
    )

    predictor.initialize_from_trained_model_folder(
        pretrained_weights_filepath, use_folds, checkpoint_name
    )

    parameters = predictor.list_of_parameters
    plans_manager = predictor.plans_manager
    network = predictor.network
    patch_size = plans_manager.get_configuration(configuration).patch_size

    return predictor, patch_size, network, parameters


def run_inference(tta_sample, model, predictor, all_tta_parameter_paths):
    save_probabilities = False

    tta_parameters = []
    for _path in all_tta_parameter_paths:
        tta_parameters.extend(torch.load(_path))

    predictor.network = deepcopy(model)
    predictor.list_of_parameters = tta_parameters
    return predict_from_data_iterator(predictor, tta_sample, save_probabilities)


def get_data_iterator_from_lists_of_filenames(
    predictor,
    input_list_of_lists: List[List[str]],
    seg_from_prev_stage_files: Union[List[str], None],
    output_filenames_truncated: Union[List[str], None],
):
    # Adapted from nnunetv2/inference/predict_from_raw_data.py
    return preprocessing_iterator_fromfiles(
        input_list_of_lists,
        seg_from_prev_stage_files,
        output_filenames_truncated,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        predictor.verbose_preprocessing,
    )


def preprocessing_iterator_fromfiles(
    list_of_lists: List[List[str]],
    list_of_segs_from_prev_stage_files: Union[None, List[str]],
    output_filenames_truncated: Union[None, List[str]],
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    verbose: bool = False,
):
    # Adapted from nnunetv2/inference/data_iterators.py
    num_samples = len(list_of_lists)
    for smp_idx in range(num_samples):
        item = preprocess_fromfile(
            list_of_lists[smp_idx],
            list_of_segs_from_prev_stage_files[smp_idx],
            output_filenames_truncated[smp_idx],
            plans_manager,
            dataset_json,
            configuration_manager,
            verbose,
        )
        yield item


def preprocess_fromfile(
    list_of_im_file: List[str],
    seg_from_prev_stage_file: str,
    output_filename_truncated: str,
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    verbose: bool = False,
):
    # Adapted from nnunetv2/inference/data_iterators.py
    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)

    data, seg, data_properties = preprocessor.run_case(
        list_of_im_file,
        seg_from_prev_stage_file,
        plans_manager,
        configuration_manager,
        dataset_json,
    )

    if seg_from_prev_stage_file is not None:
        seg_onehot = convert_labelmap_to_one_hot(
            seg[0], label_manager.foreground_labels, data.dtype
        )
        data = np.vstack((data, seg_onehot))

    data = torch.from_numpy(data).contiguous().float()

    item = {
        "data": data,
        "data_properties": data_properties,
        "ofile": output_filename_truncated,
    }
    return item



def predict_from_data_iterator(predictor,
                               sample,
                               save_probabilities: bool = False):
    # Adapted from nnunetv2/inference/predict_from_raw_data.py
    data = sample['data']
    if isinstance(data, str):
        delfile = data
        data = torch.from_numpy(np.load(data))
        os.remove(delfile)

    ofile = sample['ofile']
    if ofile is not None:
        print(f'\nPredicting {os.path.basename(ofile)}:')
    else:
        print(f'\nPredicting image of shape {data.shape}:')

    print(f'perform_everything_on_gpu: {predictor.perform_everything_on_gpu}')

    properties = sample['data_properties']
    prediction = predictor.predict_logits_from_preprocessed_data(data).cpu()

    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        prediction, predictor.plans_manager,
        predictor.configuration_manager, predictor.label_manager,
        properties, save_probabilities
    )

    compute_gaussian.cache_clear()
    # clear device cache
    empty_cache(predictor.device)
    return segmentation
