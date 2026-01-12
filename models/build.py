# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformers are Circulant Attention Learners
# Modified by Dongchen Han
# --------------------------------------------------------

from .ca_deit import ca_deit_tiny, ca_deit_small, ca_deit_base


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type in ['ca_deit_tiny', 'ca_deit_small', 'ca_deit_base']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
