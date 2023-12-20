from pathlib import Path
from copy import deepcopy

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
        return iter(())

    seg_from_prev_stage_files = [
        s if Path(s).is_file() else None for s in seg_from_prev_stage_files
    ]
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        config["num_processes"],
    )
    return data_iterator


def load_tta_data(config, dataset_raw_path, predictor):
    with suppress_stdout():
        ts_iterator = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTs",
        )
        tr_iterator = get_data_iterator(
            config,
            predictor,
            config["tta_data_filepaths"],
            dataset_raw_path,
            "imagesTr",
        )

    data = list(ts_iterator) + list(tr_iterator)

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


def run_inference(config, tta_data, model, predictor, all_tta_parameter_paths):
    save_probabilities = False
    num_processes_segmentation_export = config["num_processes"]

    tta_parameters = []
    for _path in all_tta_parameter_paths:
        tta_parameters.extend(torch.load(_path))

    predictor.network = deepcopy(model)
    predictor.list_of_parameters = tta_parameters
    predictor.predict_from_data_iterator(
        tta_data, save_probabilities, num_processes_segmentation_export
    )
