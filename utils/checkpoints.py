import warnings

import torch


def load_checkpoint(model, path, device="cpu"):
    """Load a state dict strictly, allowing only verified inactive legacy buffers."""
    state = torch.load(path, map_location=device, weights_only=True)
    expected = model.state_dict()
    discarded = []
    for index, block in enumerate(model.blocks):
        key = f"blocks.{index}.query_power_bias"
        if key in state and key not in expected:
            value = state[key]
            shape = (block.n_heads, block.pred_len, block.patch_num)
            if tuple(value.shape) != shape or torch.count_nonzero(value).item() != 0:
                raise RuntimeError(f"Unsupported nonzero or malformed legacy parameter: {key}")
            del state[key]
            discarded.append(key)
    if discarded:
        warnings.warn(
            "Ignoring inactive all-zero legacy parameters: " + ", ".join(discarded),
            UserWarning,
            stacklevel=2,
        )
    model.load_state_dict(state, strict=True)
    return model
