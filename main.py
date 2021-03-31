from src.runners.run_dataprep_pipelines import run_dataprep_pipeline
from src.utilities.make_time_of_day_analysis import time_of_day_analysis

import os
import pandas as pd

if __name__ == '__main__':
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'data'])
    figures_folder = os.sep.join([os.environ['PWD'], 'figures'])

    # set dataprep to false to read prepared data for analysis
    # set dataprep to true to run prediction and fill charge categories
    run_dataprep = True
    # default read from source to false
    read_from_source = True
    # if none, source_filename defaults to "Arrests.bz2"
    source_filename = "Arrests_-_Authorized-Access-Only_Version.csv"

    if run_dataprep:
        # set read_from_source to true if have access to source data
        # default to false runs data prep from redacted arrest data
        df = run_dataprep_pipeline(read_from_source=read_from_source, data_folder=data_folder, source_filename=source_filename)
    else:
        # set full path to target data
        filename = 'arrests_redacted_classified.bz2'
        data_file = os.sep.join([data_folder, filename])
        df = pd.read_pickle(data_file)

        filename ='arrest_clusters.csv'
        file_out = os.sep.join([data_folder, 'arrest_clusters.zip'])
        compression = dict(method='zip', archive_name=filename)
        df.to_csv(file_out, index=False, compression=compression)
        time_of_day_analysis(df, figures_folder=figures_folder)




