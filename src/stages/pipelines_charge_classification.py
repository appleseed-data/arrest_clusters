from src.utilities.config_classification import *
from src.utilities.config_dataprep import make_categorical
from src.stages.pipelines_nlp_classification import make_nlp_classification_model, apply_nlp_classification_model


def run_charge_classification(data_folder
                             , filename='arrests_redacted.bz2'
                             , nlp_model=True
                             , crosswalk='CPD_crosswalk_final.xlsx'
                             , sheet_name='CPD_crosswalk_final'):

    # logging.info('Reading prepared dataframe from', filename)
    # the target data for analysis
    data_file = os.sep.join([data_folder, filename])
    df = pd.read_pickle(data_file)
    logging.info('Preparing CPD crosswalk for macro and micro classifications.')

    # the charge description maps
    data_file = os.sep.join([data_folder, crosswalk])
    crosswalk, micro_charge_map, macro_charge_map = prep_crosswalk(filename=data_file, sheet_name=sheet_name)

    df = (df.pipe(apply_crosswalk_directmatch, micro_charge_map=micro_charge_map, macro_charge_map=macro_charge_map)
            .pipe(apply_crosswalk_fuzzymatch, micro_charge_map=micro_charge_map, macro_charge_map=macro_charge_map)
            .pipe(apply_manual_match, criteria=[('CTA - ', ['Nuisance', 'Other'])])
         )

    if nlp_model is False:
        model = make_nlp_classification_model(df, data_folder)
    else:
        model = joblib.load(model_save_path)

    df = (df.pipe(apply_nlp_classification_model, model=model, data_folder=data_folder)
            .pipe(make_categorical, cols=charge_columns_macro)
            .pipe(make_categorical, cols=charge_columns_micro)
          )

    filename = 'arrests_redacted_classified.bz2'
    data_file = os.sep.join([data_folder, filename])
    df.to_pickle(data_file)

    return df



