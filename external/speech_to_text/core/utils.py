import json
from addict import Dict
from collections import OrderedDict
import core.transforms.audio as audio
from core.transforms import AudioCompose
import core.metrics as metrics


def load_cfg(path):
    with open(path) as stream:
        cfg = Dict(json.load(stream))
    return cfg


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        elif k in source_state and v.size() != source_state[k].size():
            print(f"src: {source_state[k].size()}, tgt: {v.size()}")
            new_dict[k] = v
        else:
            print(f"key {k} not loaded...")
            new_dict[k] = v
    target.load_state_dict(new_dict)


def get_audio_transforms(cfg):
    def build_transform(name, params):
        if hasattr(audio, name):
            return getattr(audio, name)(**params)
        else:
            return None

    transforms = []
    for p in cfg:
        transform = build_transform(p["name"], p["params"])
        if transform is not None:
            transforms.append(transform)
    return AudioCompose(transforms)


def get_metrics(names):
    out = []
    for name in names:
        if hasattr(metrics, name):
            out.appen(getattr(metrics, name)())
    return out
