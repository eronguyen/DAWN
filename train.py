import os
import sys
import torch
import accelerate 
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate.utils import DistributedDataParallelKwargs
from utils.logging import setup_logging, save_config_yaml
from torch import optim

from dawn.trainer import Trainer

def main(cfg: DictConfig = None):
    # Create an instance of DistributedDataParallelKwargs and set find_unused_parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize the accelerator
    accelerator = accelerate.Accelerator(**cfg.accelerator, kwargs_handlers=[ddp_kwargs])

    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    accelerate.utils.set_seed(cfg.seed)

    setup_logging(
        is_main_process=accelerator.is_main_process,
        log_dir=cfg.trainer.save_dir,
    )

    logger = logging.getLogger(__name__)
    if accelerator.is_main_process:
        cfg_path = save_config_yaml(cfg, cfg.trainer.save_dir, filename="config.yaml")
        logger.info(f"Saved resolved config to: {cfg_path}")
    
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # # Init dataset
    dataset_args = cfg.dataset.get("args", {})
    if not cfg.dataset.train.get("_target_"):
        logger.info("Multiple training datasets found, concatenating them.")
        train_dataset = []
        for k, v in cfg.dataset.train.items():
            repeat = v.get("repeat", 1)
            logger.info(f"Building dataset {k} with weight {repeat}.")
            ds = hydra.utils.instantiate(v, **dataset_args)
            train_dataset.extend([ds] * repeat)
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        logger.info(f"Train dataset length: {len(train_dataset)}", )
    else:    
        train_dataset = hydra.utils.instantiate(cfg.dataset.train, **dataset_args)

    val_dataset = hydra.utils.instantiate(cfg.dataset.val, **dataset_args)

    # Init dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.loader.train_batch_size, 
        shuffle=True, 
        num_workers=cfg.loader.num_workers,
        pin_memory=True, 
        # prefetch_factor=1,
        
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loader.val_batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        # prefetch_factor=0,
    )

    model = hydra.utils.instantiate(cfg.model)
    
    # Optimizer

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(cfg.trainer.lr_warmup_steps * accelerator.num_processes),
        num_training_steps=(cfg.trainer.total_steps * accelerator.num_processes),
    )

    trainer = Trainer(
        cfg=cfg.trainer,
        accelerator=accelerator,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        checkpoint_path=cfg.weights,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    with hydra.initialize(config_path="configs"):
        argv = sys.argv[1:]
        print(argv)
        if len(argv) > 0 and argv[0].startswith("config="):
            config_name = argv[0].split("=")[1]
            argv = argv[1:]
        else:
            config_name = "default"
        cfg = hydra.compose(config_name=config_name, overrides=argv)
        OmegaConf.resolve(cfg)
        main(cfg)
