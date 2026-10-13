import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from .ego_dataset import EgoDataset
from .json_main_dataset import JSONMainDataset, JSONLabeledMainDataset


def __getattr__(name):
    if name == "LabeledMainDataset":
        from .main_dataset import LabeledMainDataset
        return LabeledMainDataset
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

def data_loader(data_type, config):

    if data_type == 'ego':
        dataset = EgoDataset(config.data_dir, T=config.ego_traj_len)
    elif data_type == 'main':
        from .main_dataset import LabeledMainDataset
        dataset = LabeledMainDataset(
            config.data_dir,
            config.config_path,
        )
    else:
        raise NotImplementedError(f'Unknown data type {data_type}')

    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)

__all__ = ['data_loader', 'EgoDataset', 'LabeledMainDataset', 'JSONMainDataset', 'JSONLabeledMainDataset']
