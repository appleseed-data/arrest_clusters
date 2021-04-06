from src.runners.run_dataprep_pipelines import run_dataprep_pipeline
from src.utilities.make_time_of_day_analysis import time_of_day_analysis
from src.utilities.make_geospatial_analysis import geospatial_analysis

import os
import pandas as pd

if __name__ == '__main__':
    # set path to data folder
    data_folder = os.sep.join([os.environ['PWD'], 'data'])
    figures_folder = os.sep.join([os.environ['PWD'], 'figures'])
    models_folder = os.sep.join([os.environ['PWD'], 'models'])

    source_filename = "Arrests_-_Authorized-Access-Only_Version.bz2"

    df = run_dataprep_pipeline(data_folder=data_folder
                               , models_folder=models_folder
                               , source_filename=source_filename
                               )

    time_of_day_analysis(df, data_folder=data_folder, figures_folder=figures_folder)
    # geospatial_analysis(df, data_folder=data_folder, figures_folder=figures_folder)




