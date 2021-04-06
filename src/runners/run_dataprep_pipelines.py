from src.stages.stage_dataprep import stage_dataprep
from src.stages.stage_charge_classification import stage_charge_classification
from src.utilities.config_general import *

import pandas as pd


def run_dataprep_pipeline(data_folder, models_folder, source_filename=None):

    df = stage_dataprep(data_folder, filename=source_filename)
    df = stage_charge_classification(data_folder, models_folder, filename='arrests_redacted.bz2', df=df)
    logging.info('run_dataprep_pipeline() Successfully completed data prep, ready for analysis.')

    return df