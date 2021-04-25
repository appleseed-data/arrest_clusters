"""A list of constants used for the workflow.
"""
import os
from os.path import dirname

WORKFLOW_ROOT = dirname(dirname(dirname(__file__)))
DATA_FOLDER = os.sep.join([WORKFLOW_ROOT, 'data'])
LOGGING_FOLDER = os.sep.join([WORKFLOW_ROOT, 'logs'])
FIGURES_FOLDER = os.sep.join([WORKFLOW_ROOT, 'figures'])
MODELS_FOLDER = os.sep.join([WORKFLOW_ROOT, 'models'])

ALL_FOLDERS = [DATA_FOLDER, LOGGING_FOLDER, FIGURES_FOLDER]

