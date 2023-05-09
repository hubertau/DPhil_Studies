'''CLI commands for modelling
'''

import click
from loguru import logger
import jsonlines
from pprint import PrettyPrinter
import os
from pathlib import Path
import pandas as pd

from . import *

@click.group(help='Commands relating to modelling')
@click.pass_context
def model(ctx):
    pass

@model.command()
def test():
    print('hello')

@model.command()
@click.pass_context
@click.argument('dataset')
@click.argument('checkpoint_dir')
@click.option('--num_labels', '-n', default=2, type=int)
@click.option('--init_model','-i', default='sentence-transformers/LaBSE')
@click.option('--num_epochs', '-e', default=10, type=int)
def train(ctx,
          dataset,
          checkpoint_dir,
          num_labels,
          num_epochs,
          init_model
    ):
    '''Fine tune on annotated dataset'''
    custom_trainer(
        dataset,
        checkpoint_dir = checkpoint_dir,
        init_model=init_model,
        num_labels = num_labels,
        num_train_epochs= num_epochs
    )
