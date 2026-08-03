import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.utils.collations import SparseSegmentCollationGen
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']

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
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        num_workers = self.cfg['train']['num_workers']
        loader_kwargs = {
            'batch_size': self.cfg['train']['batch_size'],
            'shuffle': shuffle,
            'num_workers': num_workers,
            'collate_fn': collate,
            'pin_memory': False,
        }
        # Worker-only options are invalid when loading synchronously in the main process.
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
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
