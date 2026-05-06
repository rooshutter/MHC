import argparse
from argparse import Namespace
from pathlib import Path
import yaml

import os
import sys

import torch
import pytorch_lightning as pl

if __name__ == "__main__":
    
    # Setup working directory and importing
    desired_directory = '/home/rhutter/MHC-Diff/'
    os.chdir(desired_directory)
    sys.path.insert(0, desired_directory)
    from model.lightning_module import Structure_Prediction_Model

    # Set seed for reproducibitliy
    seed = 42
    torch.manual_seed(seed)
    # Note: deterministic=True + benchmark=False disables cuDNN kernel tuning, slowing training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True  # Let cuDNN auto-tune for faster convolutions

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--overfit_batches', type=int, default=0, help='Number of batches to overfit on for debugging. Set to 1 to overfit on a single batch. 0 means disabled.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args_dict = args.__dict__
    for key, value in config.items():
        if isinstance(value, dict):
            args_dict[key] = Namespace(**value)
        else:
            args_dict[key] = value

    # lightning module
    model = Structure_Prediction_Model(
                args.dataset,
                args.data_dir,
                args.dataset_params,
                args.task_params,
                args.generative_model,
                args.generative_model_params,
                args.architecture,
                args.network_params,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.device,
                args.all_atom,
                run_name=args.run_name
    )

    # wandb logger
    if args.wandb_log:
        logger = pl.loggers.WandbLogger(
            save_dir=args.logdir,
            project=args.project,
            name=args.run_name,
            entity=args.entity
        ) 
    if args.all_atom:
        monitor = "error_x_val"
    else:
        monitor = "error_mol_val"


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.logdir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor=monitor,
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # setup trainer
    if args.wandb_log:
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            logger=logger,
            # callbacks=[checkpoint_callback, lr_monitor],
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            accelerator='gpu', devices=args.gpus,
            overfit_batches=args.overfit_batches,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            # logger=logger,
            # callbacks=[checkpoint_callback, lr_monitor],
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            accelerator='gpu', devices=args.gpus,
            overfit_batches=args.overfit_batches,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )

    # train
    if args.resume is False:
        trainer.fit(model)
    else:
        trainer.fit(model, ckpt_path=args.resume)