import argparse
from utils import parse_arguments, set_seed, configure_model
from models import Avatar_Generator_Model
import os
import sys
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "GAN_Font"

def train(config_file, use_wandb, run_name, run_notes):
    set_seed(32)
    config = configure_model(config_file, use_wandb)
    
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False
    
    
    xgan = Avatar_Generator_Model(config, use_wandb)
    #xgan.load_weights(config.model_path)
    xgan.train()


if __name__ == '__main__':
    args = parse_arguments()
    use_wandb = args.wandb
    run_name = args.run_name
    run_notes = args.run_notes

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)
