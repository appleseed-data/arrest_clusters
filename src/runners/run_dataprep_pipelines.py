from src.stages.stage_dataprep import stage_dataprep
from src.stages.stage_charge_classification import stage_charge_classification
from src.utilities.config import Config


def run_dataprep_pipeline(data_folder, models_folder, source_filename=None, primary_file_name='arrests_redacted.bz2'):

    df = stage_dataprep(data_folder, input_file=source_filename, output_file=primary_file_name)
    df = stage_charge_classification(data_folder=data_folder, models_folder=models_folder, input_file=primary_file_name, df=df)
    Config.my_logger.info('run_dataprep_pipeline() Successfully completed data prep, ready for analysis.')

    return df
