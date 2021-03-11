from src.utilities.config_classification import *
from src.utilities.config_dataprep import make_categorical
from src.utilities.pipelines_nlp_classification import make_nlp_classification_model, apply_nlp_classification_model


def run_charge_classification(filename='data/arrests_redacted.bz2'
                             , nlp_model=True):

    # logging.info('Reading prepared dataframe from', filename)
    # the target data for analysis
    df = pd.read_pickle(filename)
    logging.info('Preparing CPD crosswalk for macro and micro classifications.')

    # the charge description maps
    crosswalk, micro_charge_map, macro_charge_map = prep_crosswalk(filename='data/CPD_crosswalk_final.xlsx', sheet_name='CPD_crosswalk_final')

    df = (df.pipe(apply_crosswalk_directmatch, micro_charge_map=micro_charge_map, macro_charge_map=macro_charge_map)
            .pipe(apply_crosswalk_fuzzymatch, micro_charge_map=micro_charge_map, macro_charge_map=macro_charge_map)
            .pipe(apply_manual_match, criteria=[('CTA - ', ['Nuisance', 'Other'])])
         )

    if nlp_model is False:
        model = make_nlp_classification_model(df)
    else:
        model = joblib.load(model_save_path)

    df = (df.pipe(apply_nlp_classification_model, model=model)
            .pipe(make_categorical, cols=charge_columns_macro)
            .pipe(make_categorical, cols=charge_columns_micro)
          )

    df.to_pickle('data/arrests_redacted_classified.bz2')

    return df



