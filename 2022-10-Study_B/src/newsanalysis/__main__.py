from .dataviz import cli_dataviz
from .data_utils import cli_data_utils, mediacloud_collect
from .modelling import cli_modelling
import click
from loguru import logger
import sys
import logging

# intercept default logging package from submodules (e.g. SentenceTrasnformers)
# from recipe in loguru: https://loguru.readthedocs.io/en/stable/overview.html
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0)
sys.stdout = sys.stderr #to capture print statements

@click.group()
@click.pass_context
@click.option('--debug/--no-debug', default=False)
@click.option('--gpu/--no-gpu', default=False)
@click.option('--dask/--no-dask', default=False)
@click.option('--log_file')
def cli(ctx, debug, gpu, dask, log_file):
    """News Analysis package.

    """
    # logger.info(f"Debug mode is {'on' if debug else 'off'}")
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=True)
    if log_file:
        logger.add(log_file, backtrace=True, diagnose=True)

    ctx.obj = {}
    ctx.obj['DEBUG'] = debug
    ctx.obj['GPU'] = gpu
    ctx.obj['DASK'] = dask

cli.add_command(cli_dataviz.viz)
cli.add_command(cli_data_utils.preprocess)
cli.add_command(cli_data_utils.postprocess)
cli.add_command(cli_modelling.model)
cli.add_command(mediacloud_collect.mediac)

if __name__ == '__main__':
    cli()