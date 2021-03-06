from src.utilities.make_classification import *
from src.utilities.constants import *
from src.utilities.make_dataprep import make_categorical
from src.utilities.make_nlp_classifications import apply_nlp_classification_model_charge_descriptions, apply_nlp_match_police_related

import os
import logging


def stage_charge_classification(
                                  input_file='arrests_redacted.bz2'
                                , df=None
                                , output_file='arrests_redacted_classified.bz2'
                                , model_name_charge_classification='arrest_charge_classification'
                                , crosswalk='CPD_crosswalk_final.xlsx'
                                , sheet_name='CPD_crosswalk_final'
                                , output_filename='arrest_clusters'
                                ):

    logging.info('stage_charge_classification() Starting charge classification pipeline.')
    # the target data for analysis
    input_file_path = os.sep.join([DATA_FOLDER, input_file])
    output_file = os.sep.join([DATA_FOLDER, output_file])

    if os.path.exists(output_file):
        logging.info(f'Found exiting processed file at {output_file}, returning file from disk.')
        logging.info(f'If you want to run the classification pipeline, delete or rename the input file located at {input_file_path}')
        df = pd.read_pickle(output_file)
        return df

    elif df is not None:
        logging.info('Continuing pipeline with dataframe.')
        df = df

    elif os.path.exists(input_file_path):
        logging.info(f'Did not find existing output file or dataframe from pipeline, reading from input file instead from disk at {input_file_path}')
        df = pd.read_pickle(input_file_path)

    logging.info('Preparing CPD crosswalk for macro and micro classifications.')
    # the charge description maps
    crosswalk_file = os.sep.join([DATA_FOLDER, crosswalk])
    crosswalk, micro_charge_map, macro_charge_map, police_related_map = prep_crosswalk(filename=crosswalk_file, sheet_name=sheet_name)

    df = (df.pipe(apply_crosswalk_directmatch
                  , micro_charge_map=micro_charge_map
                  , macro_charge_map=macro_charge_map
                  , police_related_map=police_related_map
                  )
            .pipe(apply_crosswalk_fuzzymatch
                  , micro_charge_map=micro_charge_map
                  , macro_charge_map=macro_charge_map
                  , police_related_map=police_related_map
                  )
            .pipe(apply_manual_match, criteria=[('CTA - ', ['Nuisance', 'Other'])])
         )

    logging.info('Classifying remaining charge description records with NLP models.')

    df = (df.pipe(apply_nlp_classification_model_charge_descriptions
                  , model_name_charge_classification=model_name_charge_classification)
            .pipe(make_categorical, cols=Config.charge_columns_macro)
            .pipe(make_categorical, cols=Config.charge_columns_micro)
            .pipe(apply_nlp_match_police_related)
          )

    logging.info(f'Writing processed file from data prep pipeline to {output_file}')
    df.to_pickle(output_file, protocol=2)

    # write data as zipped csv for convenience and sharint
    output_csv = output_filename + '.csv'
    output_zip_shell = output_filename + '.zip'
    output_csv_path = os.sep.join([DATA_FOLDER, output_zip_shell])
    compression = dict(method='zip', archive_name=output_csv)
    df.to_csv(output_csv_path, index=False, compression=compression)

    return df



