from src.utilities.make_dataprep import *
from src.utilities.config import Config

def stage_dataprep(data_folder, input_file, output_file):
    Config.my_logger.info('stage_dataprep() --Starting arrest data pipeline')
    pd.set_option('display.max_columns', None)

    input_file_path = os.sep.join([data_folder, input_file])
    Config.my_logger.info(f'Checking Data at {input_file_path}')

    output_file_path = os.sep.join([data_folder, output_file])

    if os.path.exists(output_file_path):
        Config.my_logger.info(f'Found exiting output file, returning processed file from disk found at {output_file_path}.')
        Config.my_logger.info(f'To run the pipeline from source file, delete or rename the input file at {output_file_path}')
        df = pd.read_pickle(output_file_path)
        return df

    elif os.path.exists(input_file_path):
        Config.my_logger.info(f'Could not locate processed arrest data at {output_file_path}, proceeding to data prep pipeline.')
        Config.my_logger.info(f'Found Source Data, starting pipeline from source at {input_file_path}')
        df = pd.read_pickle(input_file_path)
        Config.my_logger.info(f'{input_file_path} has {len(df)} records')

        df = (df.pipe(parse_cols)
                .pipe(optimize
                      , special_mappings={'string': Config.charge_columns
                                          ,'datetime': ['received_in_lockup', 'released_from_lockup']}
                      , parse_col_names=False
                      , enable_mp=True
                      )
                # commented out becuase names are not being processed in this pipeline currently
                # .pipe(make_titlecase, cols=['first_name', 'last_name', 'middle_name'])
                .pipe(make_redact, cols=Config.redact_columns)
                .pipe(make_arrest_year_month)
                .pipe(categorize_charge_cols)
                .pipe(prep_districts)
                .pipe(prep_beats, data_folder=data_folder)
                .pipe(prep_time_of_day)
              )

        # drop unnecessary columns
        df = df.drop(columns=Config.drop_columns)

        Config.my_logger.info('Write prepared and redacted dataframe to pickle')

        df.to_pickle(output_file_path, protocol=2)
        Config.my_logger.info(f'--Completed arrest data pipeline. Write dataframe to {output_file_path}')
        Config.my_logger.info('To run analysis, re-run and set data_prep read from source to False')

        return df

