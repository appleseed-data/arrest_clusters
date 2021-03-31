from src.utilities.config_dataprep import *

def run_pipeline_from_source(data_folder, filename='Arrests.bz2'):
    logging.info('--Starting arrest data pipeline')
    pd.set_option('display.max_columns', None)

    data_file = os.sep.join([data_folder, filename])

    logging.info(f'Reading Data from {data_file}')

    df = pd.read_pickle(data_file)

    logging.info(f'{data_file} has {len(df)} records')

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
            .pipe(make_arrest_year_month)
            .pipe(categorize_charge_cols)
          )

    # drop unnecessary columns
    df = df.drop(columns=['charges_statute', 'charges_description', 'charges_type', 'charges_class', 'charges_fbi_code'])

    logging.info('Write prepared and redacted dataframe to pickle')
    filename='arrests_redacted.bz2'
    data_file = os.sep.join([data_folder, filename])
    df.to_pickle(data_file)
    logging.info('--Completed arrest data pipeline')
    logging.info('To run analysis, re-run and set data_prep read from source to False')

    return df

