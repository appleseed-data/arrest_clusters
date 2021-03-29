from src.utilities.pipelines_dataprep import run_pipeline_from_source
from src.utilities.pipelines_charge_classification import run_charge_classification
from src.utilities.config_general import *

import pandas as pd


def run_dataprep_pipeline(data_folder, read_from_source=False, source_filename=None):
    # read from source if source csv is from chicago data portal
    # if false, read from redacted file (public facing)
    if read_from_source:
        if source_filename is not None:
            filename = source_filename
            logging.info(f'Reading source from {filename}')
            # if source is csv, compress and write to file
            if '.csv' in source_filename:
                logging.info(f'Converting file to compressed pickle file.')
                file_path = os.sep.join([data_folder, source_filename])
                df = pd.read_csv(file_path)
                source_base = os.path.splitext(source_filename)[0]
                source_filename = f'{source_base}.bz2'
                file_path = os.sep.join([data_folder, source_filename])
                df.to_pickle(file_path, protocol=2, compression='infer')
                logging.info(f'Wrote compressed file to {file_path}.')
                logging.info(f"Filename is now {filename}")
                filename = source_filename

            elif '.bz2' in source_filename:
                file_path = os.sep.join([data_folder, source_filename])
                logging.info(f'Reading compressed file from {file_path}.')
                filename = source_filename

        else:
            filename = 'Arrests.bz2'
        df = run_pipeline_from_source(data_folder, filename=filename)
    else:
        df = run_charge_classification(data_folder, filename='arrests_redacted.bz2')

    return df