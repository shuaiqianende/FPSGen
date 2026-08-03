import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fpsgen.datasets.datasets as datasets
import fpsgen.models.gen_student as models

def set_deterministic():
    """Seed the stochastic training components for repeatable experiments."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(dirname(abspath(__file__))),'configs/train_student.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.point_cloud). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-point_cloud',
              type=str,
              help='path to checkpoint file (.point_cloud) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, weights, checkpoint, test):
    """Launch stage-3 PointFlow training, checkpoint resume, or evaluation."""
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    print('cfg', cfg)
    # Keep dataset locations out of tracked configuration files.
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')

    if weights is None:
        model = models.DiffusionPoints(cfg)
    else:
        model = models.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
        print(model.hparams)

    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1
                                 )

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError('FPSGen training requires a CUDA-visible GPU. Set CUDA_VISIBLE_DEVICES correctly.')
    requested_gpus = min(cfg['train']['n_gpus'], visible_gpus)
    if requested_gpus > 1:
        # Use synchronized sparse batch normalization when training with DDP.
        cfg['train']['n_gpus'] = requested_gpus
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(gpus=requested_gpus,
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs=cfg['train']['max_epoch'],
                          limit_train_batches=cfg['train'].get('limit_train_batches', 1.0),
                          limit_test_batches=cfg['train'].get('limit_test_batches', 1.0),
                          callbacks=[lr_monitor, checkpoint_saver],
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=0,
                          limit_val_batches=0.002,
                          # Lightning 1.8 selects DDP through ``strategy``.
                          strategy='ddp',
                          )
    else:
        trainer = Trainer(gpus=1,
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs=cfg['train']['max_epoch'],
                          limit_train_batches=cfg['train'].get('limit_train_batches', 1.0),
                          limit_test_batches=cfg['train'].get('limit_test_batches', 1.0),
                          callbacks=[lr_monitor, checkpoint_saver],
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=0,
                          limit_val_batches=0.002,
                          )

    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
