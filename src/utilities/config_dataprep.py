from src.utilities.config_general import *

# import modin.pandas as pd
# import ray
# ray.init()
import pandas as pd
import re
import numpy as np

charge_columns = ['charge_1_description'
                 , 'charge_2_description'
                 , 'charge_3_description'
                 , 'charge_4_description'
                 ]
def make_categorical(df, cols):
    df[cols] = df[cols].astype('category')
    return df

def parse_cols(df):
    logging.info('Parsing column headers to lower case and replacing spaces with underscore.')
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    return df

def make_titlecase(df, cols):
    logging.info(f'Converting data to titlecase and removing punctiontion for columns:\n{cols}')
    def make_titlecase_(x):
        x = x.str.title()
        return x

    def strip_punctuation_(x):
        x = x.str.strip()
        x = x.str.replace('\.', '', regex=False)
        x = x.str.replace('\'', '', regex=False)
        x = x.str.replace('\,', '', regex=False)
        x = x.str.replace('\s+', ' ', regex=False)
        x = x.str.title()
        return x

    df[cols] = df[cols].astype('object')
    df[cols] = df[cols].apply(lambda x: make_titlecase_(x))
    df[cols] = df[cols].apply(lambda x: strip_punctuation_(x))
    df[cols] = df[cols].astype('category')

    return df

def reduce_precision(df, charge_cols=None):
    """
    :param df: attempts to auto-optimize a dataframe by applying least precision to each col type
    :return: the same dataframe but with different col dtypes, if applicable
    """
    logging.info(f'Starting DataFrame Optimization. Starting with {mem_usage(df)} memory.')
    cols_to_convert = []
    date_strings = ['_date', 'date_', 'date']

    for col in df.columns:
        col_type = df[col].dtype
        if 'string' not in col_type.name and col_type.name != 'category' and 'datetime' not in col_type.name:
            cols_to_convert.append(col)

    # leave charge columns untouched to synch with crosswalk mapping
    if charge_cols is not None:
        cols_to_convert = [x for x in cols_to_convert if x not in charge_cols]
        df[charge_cols] = df[charge_cols].astype('string')

    def _reduce_precision(x):
        col_type = x.dtype
        unique_data = list(x.unique())
        bools = [True, False, 'true', 'True', 'False', 'false']
        #TODO: account for only T or only F or 1/0 situations
        n_unique = float(len(unique_data))
        n_records = float(len(x))
        cat_ratio = n_unique / n_records

        try:
            unique_data.remove(np.nan)
        except:
            pass

        if 'int' in str(col_type):
            c_min = x.min()
            c_max = x.max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                x= x.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                x = x.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                x = x.astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                x = x.astype(np.int64)

                # TODO: set precision to unsigned integers with nullable NA

        elif 'float' in str(col_type):
            c_min = x.min()
            c_max = x.max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    x = x.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                x = x.astype(np.float32)
            else:
                x = x.astype(np.float64)

        elif 'datetime' in col_type.name or any(i in str(x.name).lower() for i in date_strings):
            try:
                x = pd.to_datetime(x)
            except:
                pass

        elif any(i in bools for i in unique_data):
            x = x.astype('boolean')
            #TODO: set precision to bool if boolean not needed

        elif cat_ratio < .1 or n_unique < 20:
            try:
                x = x.str.title()
            except:
                pass

            x = pd.Categorical(x)

        elif all(isinstance(i, str) for i in unique_data):
            x = x.astype('string')

        return x

    df[cols_to_convert] = df[cols_to_convert].apply(lambda x: _reduce_precision(x))


    logging.info(f'Converted DF with new dtypes as follows:\n{df.dtypes}')
    logging.info(f'Completed DataFrame Optimization. Ending with {mem_usage(df)} memory.')

    return df

def mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        source: https://gist.github.com/enamoria/fa9baa906f23d1636c002e7186516a7b
    """
    mem = df.memory_usage().sum() / 1024 ** 2
    return '{:.2f} MB'.format(mem)

def make_redact(df, cols):
    df = df.drop(columns=cols)
    return df