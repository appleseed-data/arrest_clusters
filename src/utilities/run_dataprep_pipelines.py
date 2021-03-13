from src.utilities.pipelines_dataprep import run_pipeline_from_source
from src.utilities.pipelines_charge_classification import run_charge_classification

def run_dataprep_pipeline(data_folder, read_from_source=False):
    # read from source if source csv is from chicago data portal
    # if false, read from redacted file (public facing)
    if read_from_source:
        df = run_pipeline_from_source(data_folder)
    else:
        df = run_charge_classification(data_folder)

    return df