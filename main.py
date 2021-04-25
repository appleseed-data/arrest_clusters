from src.runners.run_dataprep_pipelines import run_dataprep_pipeline
from src.runners.run_analysis_pipelines import run_analysis_pipeline
from src.utilities.config import run_configuration


if __name__ == '__main__':

    run_configuration()

    source_filename = "Arrests_-_Authorized-Access-Only_Version.bz2"

    df = run_dataprep_pipeline(source_filename=source_filename)

    run_analysis_pipeline(df=df)

