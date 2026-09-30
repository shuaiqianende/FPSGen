import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.utils.collations import SparseSegmentCollationGen
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']


def deterministic_worker_init(worker_id):
    """Seed one DataLoader worker without inheriting CPU thread pools.

    ``torch.initial_seed`` is set by DataLoader from its (optionally seeded)
    generator and is unique per worker.  Keeping this helper at module scope
    makes it picklable for both fork and spawn worker start methods.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # Eight workers each using a large OpenMP pool makes point preprocessing
    # contend with the training process.  This is enabled only by research
    # configs that opt into deterministic workers.
    torch.set_num_threads(1)

class TemporalKittiDataModule(LightningDataModule):
    """Lightning data module for full-sequence SemanticKITTI training frames."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def _dataloader(self, sequences, split, shuffle):
        collate = SparseSegmentCollationGen()
        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=sequences,
            split=split,
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'],
            gt_dir=self.cfg['data'].get('gt_dir', 'gt_'))
        num_workers = self.cfg['train']['num_workers']
        loader_kwargs = {
            'batch_size': self.cfg['train']['batch_size'],
            'shuffle': shuffle,
            'num_workers': num_workers,
            'collate_fn': collate,
            # Historical configs retain pageable host tensors.  Research speed
            # configs can opt into pinned memory for asynchronous H2D copies.
            'pin_memory': self.cfg['train'].get('pin_memory', False),
        }
        if self.cfg['train'].get('deterministic_worker_init', False):
            generator = torch.Generator()
            generator.manual_seed(int(self.cfg['train'].get('dataloader_seed', 42)))
            loader_kwargs.update(
                worker_init_fn=deterministic_worker_init,
                generator=generator,
            )
        # Worker-only options are invalid when loading synchronously in the main process.
        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=self.cfg['train'].get('persistent_workers', True),
                prefetch_factor=self.cfg['train'].get('prefetch_factor', 2),
            )
        return DataLoader(data_set, **loader_kwargs)

    def train_dataloader(self):
        """Create the shuffled sparse batch loader used by all training stages."""
        return self._dataloader(
            self.cfg['data']['train'], self.cfg['data']['split'], shuffle=True
        )

    def test_dataloader(self):
        """Create a deterministic loader for Lightning's ``trainer.test`` path."""
        data_cfg = self.cfg['data']
        sequences = data_cfg.get('test') or data_cfg.get('validation') or data_cfg['train']
        return self._dataloader(sequences, 'test', shuffle=False)

dataloaders = {
    'KITTI': TemporalKittiDataModule,
}
