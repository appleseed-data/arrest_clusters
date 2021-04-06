from src.utilities.config_classification import *
from src.utilities.config_dataprep import make_categorical
from src.utilities.make_nlp_classifications import make_nlp_classification_model_charge_descriptions, apply_nlp_classification_model_charge_descriptions, apply_nlp_match_police_related


def stage_charge_classification(data_folder
                                , models_folder
                                , filename='arrests_redacted.bz2'
                                , df=None
                                , output_file='arrests_redacted_classified.bz2'
                                , nlp_model=True
                                , crosswalk='CPD_crosswalk_final.xlsx'
                                , sheet_name='CPD_crosswalk_final'):

    logging.info('stage_charge_classification() Starting charge classification pipeline.')
    # the target data for analysis
    input_file = os.sep.join([data_folder, filename])
    output_file = os.sep.join([data_folder, output_file])

    if os.path.exists(output_file):
        logging.info(f'Found exiting processed file at {output_file}, returning file from disk.')
        logging.info(f'If you want to run the classification pipeline, delete or rename the input file located at {input_file}')
        df = pd.read_pickle(output_file)
        return df

    elif df is not None:
        logging.info('Continuing pipeline with dataframe.')
        df = df

    elif os.path.exists(input_file):
        logging.info(f'Did not find existing output file or dataframe from pipeline, reading from input file instead from disk at {input_file}')
        df = pd.read_pickle(input_file)

    logging.info('Preparing CPD crosswalk for macro and micro classifications.')
    # the charge description maps
    crosswalk_file = os.sep.join([data_folder, crosswalk])
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

    if nlp_model is False:
        model = make_nlp_classification_model_charge_descriptions(df, data_folder)
    else:
        model = joblib.load(model_save_path)

    df = (df.pipe(apply_nlp_classification_model_charge_descriptions, model=model, data_folder=data_folder)
            .pipe(make_categorical, cols=charge_columns_macro)
            .pipe(make_categorical, cols=charge_columns_micro)
            .pipe(apply_nlp_match_police_related, data_folder=data_folder, models_folder=models_folder)
          )
    logging.info(f'Writing processed file from data prep pipeline to {output_file}')
    df.to_pickle(output_file)

    # write data as zipped csv for convenience and sharint
    output_csv ='arrest_clusters.csv'
    output_csv_path = os.sep.join([data_folder, 'arrest_clusters.zip'])
    compression = dict(method='zip', archive_name=output_csv)
    df.to_csv(output_csv_path, index=False, compression=compression)

    return df



