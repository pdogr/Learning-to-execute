import argparse
import collections
import os
from pathlib import Path
from random import choice
from matplotlib.pyplot import xticks

import numpy as np
import torch
from torchtext.data.iterator import batch

import models
from base import get_optimizer
from dataloader import Dataloader
from parse_config import ConfigParser
from trainer import Trainer
from utils.loss import AvgPerplexity
from utils.metric import Accuracy
from utils.constants import EOS_TOK, PAD_TOK, SOS_TOK
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(config, difficulty,type):
    logger = config.get_logger('train')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    batch_size = config['data_loader']['args']['batch_size']
    tgt_preprocessing = None
    src_preprocessing = None
    train_loader = Dataloader(
        device=device, difficulty=difficulty,type=type,
        src_preprocessing=src_preprocessing, tgt_preprocessing=tgt_preprocessing,
        batch_size=batch_size)

    valid_loader = Dataloader(
        device=device, difficulty=difficulty,type=type,
        src_preprocessing=src_preprocessing, tgt_preprocessing=tgt_preprocessing,
        batch_size=batch_size, train=False)
    
    model_args = config['arch']['args']
    model_args.update({
        'src_vocab': train_loader.src_vocab,
        'tgt_vocab': train_loader.tgt_vocab,
        'sos_tok': SOS_TOK,
        'eos_tok': EOS_TOK,
        'pad_tok': PAD_TOK,
        'device': device
    })
    model = getattr(models, config['arch']['type'])(**model_args)
    weight = torch.ones(len(train_loader.tgt_vocab))
    criterion = AvgPerplexity(
        ignore_idx=train_loader.tgt_vocab.stoi[PAD_TOK],
        weight=weight)

    criterion.to(device)

    optimizer = get_optimizer(
        optimizer_params=filter(
            lambda p: p.requires_grad, model.parameters()),
        args_dict=config['optimizer'])

    metrics_ftns = [Accuracy(
        train_loader.tgt_vocab.stoi[PAD_TOK])]
    # for param in model.parameters():
    #     param.data.uniform_(-0.08, 0.08)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_ftns=metrics_ftns,
        optimizer=optimizer,
        config=config,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        log_step=1, len_epoch=200
    )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '-c',
        '--config',
        default=None,
        type=str,
        help='config file path (default: None)')
    args.add_argument(
        '-r',
        '--resume',
        default=None,
        type=str,
        help='path to latest checkpoint (default: None)')
    args.add_argument(
        '-d',
        '--difficulty',
        type=int,
        help='Change difficulty',
        required=True)
    args.add_argument(
        '-t',
        '--type',
        type=str,
        help='Change type (choice)',choices=['inverse','add'],
        required=True)
    args.add_argument(
        '-de',
        '--device',
        default=None,
        type=str,
        help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    options = [
        CustomArgs(
            ['--lr', '--learning-rate'],
            type=float,
            target='optimizer;optimizer;args;lr',
            help='Change optimzer learning_rate'),
        CustomArgs(
            ['--bs', '--batch-size'],
            type=int,
            target='data_loader;args;batch_size',
            help='Change batch_size of dataloader')
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args.difficulty, args.type)
