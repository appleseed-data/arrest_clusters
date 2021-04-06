from src.utilities.config_dataprep import *
from src.utilities.config import Config

def stage_dataprep(data_folder, filename):
    Config.my_logger.info('stage_dataprep() --Starting arrest data pipeline')
    pd.set_option('display.max_columns', None)

    input_file = os.sep.join([data_folder, filename])
    Config.my_logger.info(f'Checking Data at {input_file}')

    output_file = 'arrests_redacted.bz2'
    output_file = os.sep.join([data_folder, output_file])

    if os.path.exists(output_file):
        Config.my_logger.info(f'Found exiting output file, returning processed file from disk found at {output_file}.')
        Config.my_logger.info(f'To run the pipeline from source file, delete or rename the input file at {input_file}')
        df = pd.read_pickle(output_file)
        return df

    elif os.path.exists(input_file):
        Config.my_logger.info(f'Could not locate processed arrest data at {output_file}, proceeding to data prep pipeline.')
        Config.my_logger.info(f'Found Source Data, starting pipeline from source at {input_file}')
        df = pd.read_pickle(input_file)
        Config.my_logger.info(f'{input_file} has {len(df)} records')

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
                .pipe(reduce_precision, charge_cols=Config.charge_columns)
                # commented out becuase names are not being processed in this pipeline currently
                # .pipe(make_titlecase, cols=['first_name', 'last_name', 'middle_name'])
                .pipe(make_redact, cols=redact_columns)
                .pipe(make_arrest_year_month)
                .pipe(categorize_charge_cols)
                .pipe(prep_districts)
                .pipe(prep_beats, data_folder=data_folder)
                .pipe(prep_time_of_day)
              )

        # drop unnecessary columns
        df = df.drop(columns=['charges_statute', 'charges_description', 'charges_type', 'charges_class', 'charges_fbi_code'])

        Config.my_logger.info('Write prepared and redacted dataframe to pickle')

        df.to_pickle(output_file)
        Config.my_logger.info(f'--Completed arrest data pipeline. Write dataframe to {output_file}')
        Config.my_logger.info('To run analysis, re-run and set data_prep read from source to False')

        return df

