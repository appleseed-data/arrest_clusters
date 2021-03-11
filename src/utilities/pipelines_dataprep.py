from src.utilities.config_dataprep import *

def run_pipeline_from_source(filename='data/Arrests.bz2'):
    logging.info('--Starting arrest data pipeline')
    pd.set_option('display.max_columns', None)

    logging.info(f'Reading Data from {filename}')

    df = pd.read_pickle(filename)

    redact_columns = ['first_name'
                    , 'last_name'
                    , 'middle_name'
                    , 'cb_no'
                    , 'case_number'
                    , 'street_no'
                    , 'street_dir'
                    , 'street_name'
                    ]

    df = (df.pipe(parse_cols)
            .pipe(reduce_precision, charge_cols=charge_columns)
            # .pipe(make_titlecase, cols=['first_name', 'last_name', 'middle_name'])
            .pipe(make_redact, cols=redact_columns)
          )

    # drop unnecessary columns
    df = df.drop(columns=['charges_statute', 'charges_description', 'charges_type', 'charges_class', 'charges_fbi_code'])

    logging.info('Write prepared and redacted dataframe to pickle')
    df.to_pickle('data/arrests_redacted.bz2')
    logging.info('--Completed arrest data pipeline')

    return df

