from src.utilities.run_dataprep_pipelines import run_dataprep_pipeline
from src.utilities.make_figures import time_of_day_analysis


import pandas as pd

if __name__ == '__main__':

    # set dataprep to false to read prepared data for analysis
    # set dataprep to true to run prediction and fill charge categories
    run_dataprep = False

    if run_dataprep:
        # set read_from_source to true if have access to source data
        # default to false
        df = run_dataprep_pipeline(read_from_source=False)
    else:
        df = pd.read_pickle('data/arrests_redacted_classified.bz2')


    time_of_day_analysis(df)




