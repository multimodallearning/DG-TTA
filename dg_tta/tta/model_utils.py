from copy import deepcopy

from torch._dynamo import OptimizedModule

from dg_tta.tta.torch_utils import (
    register_forward_pre_hook_at_beginning,
    register_forward_hook_at_beginning,
    hookify,
)


def get_model_from_network(network, modifier_fn_module, parameters=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    # Register hook that modifies the input prior to custom augmentation
    modify_tta_input_fn = modifier_fn_module.ModifierFunctions.modify_tta_input_fn
    register_forward_pre_hook_at_beginning(
        model, hookify(modify_tta_input_fn, "forward_pre_hook")
    )

    # Register hook that modifies the output of the model
    modfify_tta_model_output_fn = (
        modifier_fn_module.ModifierFunctions.modfify_tta_model_output_fn
    )
    register_forward_hook_at_beginning(
        model, hookify(modfify_tta_model_output_fn, "forward_hook")
    )

    return model


running_stats_buffer = {}


def buffer_running_stats(m):
    _id = id(m)
    if (
        hasattr(m, "running_mean")
        and hasattr(m, "running_var")
        and not _id in running_stats_buffer
    ):
        if m.running_mean is not None and m.running_var is not None:
            running_stats_buffer[_id] = [m.running_mean.data, m.running_var.data]


def apply_running_stats(m):
    _id = id(m)
    if (
        hasattr(m, "running_mean")
        and hasattr(m, "running_var")
        and _id in running_stats_buffer
    ):
        m.running_mean.data.copy_(other=running_stats_buffer[_id][0])
        m.running_var.data.copy_(
            other=running_stats_buffer[_id][1]
        )  # Copy into .data to prevent backprop errors
        del running_stats_buffer[_id]
