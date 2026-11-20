# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    hier = _hierarchical_block_offsets(model)
    if hasattr(model, 'blocks'):
        num_layers = len(model.blocks) + 1
    elif hier is not None:
        num_layers = hier[1] + 1
    elif not hasattr(model, 'layer1'):
        # no single block stack (e.g. the EfficientNet+Swin hybrid): one LR for every parameter
        print("[lr_decay] no block structure found -> layer_decay ignored (uniform LR)")
        num_layers = 0
        layer_decay = 1.0
    else:
        # use the number of layers in the ResNet model as a default value
        num_layers = len(model.layer1) + len(model.layer2) + len(model.layer3) + len(model.layer4) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        # (timm Swin lists module names, e.g. '...attn.cpb_mlp', so also match by prefix)
        if p.ndim == 1 or n in no_weight_decay_list or n.endswith('logit_scale') \
                or any(n.startswith(m + '.') for m in no_weight_decay_list):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        if hier is not None and not hasattr(model, 'blocks'):
            layer_id = get_layer_id_for_hierarchical(n, num_layers, hier[0])
        else:
            layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def _hierarchical_block_offsets(model):
    """For timm hierarchical models (Swin: layers.i.blocks.j, ConvNeXt: stages.i.blocks.j).
    Returns (stage_prefix, offsets, total_blocks) where offsets[i] = #blocks before stage i."""
    for attr in ('layers', 'stages'):
        stages = getattr(model, attr, None)
        if stages is not None and all(hasattr(s, 'blocks') for s in stages):
            offsets, total = [], 0
            for s in stages:
                offsets.append(total)
                total += len(s.blocks)
            return (attr, offsets), total
    return None


def get_layer_id_for_hierarchical(name, num_layers, prefix_offsets):
    """Blocks are numbered consecutively across stages (1..total); stem/patch_embed = 0;
    a stage's downsample shares the id of the first block of that stage; norm/head = num_layers."""
    attr, offsets = prefix_offsets
    parts = name.split('.')
    if parts[0] in ('patch_embed', 'stem', 'absolute_pos_embed'):
        return 0
    if parts[0] == attr:
        i = int(parts[1])
        if parts[2] == 'blocks':
            return offsets[i] + int(parts[3]) + 1
        return offsets[i] + 1  # downsample / other stage-level params
    return num_layers


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers