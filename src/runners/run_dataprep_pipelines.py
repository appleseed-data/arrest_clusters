from src.stages.pipelines_dataprep import run_pipeline_from_source
from src.stages.pipelines_charge_classification import run_charge_classification
from src.utilities.config_general import *

import pandas as pd


def run_dataprep_pipeline(data_folder, models_folder, source_filename=None):

    df = run_pipeline_from_source(data_folder, filename=source_filename)
    df = run_charge_classification(data_folder, models_folder, filename='arrests_redacted.bz2')
    logging.info('run_dataprep_pipeline() Successfully completed data prep, ready for analysis.')

    return df