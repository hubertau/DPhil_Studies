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
def train(ctx, dataset, checkpoint_dir):
    '''Fine tune on annotated dataset'''
    custom_trainer(
        dataset,
        checkpoint_dir = checkpoint_dir
    )
