from src.stages.stage_dataprep import stage_dataprep
from src.stages.stage_charge_classification import stage_charge_classification
from src.utilities.config import Config
import logging


def run_dataprep_pipeline(source_filename=None, primary_file_name='arrests_redacted.bz2'):

    df = stage_dataprep(input_file=source_filename, output_file=primary_file_name)
    df = stage_charge_classification(input_file=primary_file_name, df=df)
    logging.info('run_dataprep_pipeline() Successfully completed data prep, ready for analysis.')

    return df
