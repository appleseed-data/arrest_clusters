from src.utilities.config import Config

# import modin.pandas as pd
# import ray
# ray.init()
import pandas as pd
import re
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import logging

def prep_districts(df, target_col='district'):
    Config.my_logger.info('prep_districts() Converting district data to string.')
    df[target_col] = df[target_col].fillna(0)
    df[target_col] = df[target_col].astype('int')
    df[target_col] = df[target_col].astype('str')
    return df


def prep_beats(df, data_folder, target_col='beat', crosswalk_file='cpd_units_beats_crosswalk.csv'):
    Config.my_logger.info('prep_beats() Converting beats and unit data to string')
    df[target_col] = df[target_col].fillna(0)
    df[target_col] = df[target_col].astype('int')
    df[target_col] = df[target_col].astype('str')
    df[target_col] = df[target_col].str.zfill(4)

    Config.my_logger.info(f'Reading unit beat crosswalk from {crosswalk_file}')
    data_path = os.sep.join([data_folder, crosswalk_file])
    cpd_unit_beat_crosswalk = pd.read_csv(data_path)
    cpd_unit_beat_crosswalk = cpd_unit_beat_crosswalk.astype('str')
    cpd_unit_beat_crosswalk['unit'] = cpd_unit_beat_crosswalk['unit'].str.zfill(3)
    cpd_unit_beat_crosswalk[target_col] = cpd_unit_beat_crosswalk[target_col].str.zfill(4)

    mapper = cpd_unit_beat_crosswalk.set_index('beat')
    mapper = mapper.to_dict()['unit']
    df['unit'] = df[target_col].map(mapper)

    return df


def prep_time_of_day(df, tgt_date_col='arrest_date', index_id='arrest_id'):
    """
    break out arrest times into consumable integers for grouping
    # ref: https://stackoverflow.com/questions/32344533/how-do-i-round-datetime-column-to-nearest-quarter-hour
    """
    Config.my_logger.info('prep_time_of_day() extracting time of day columns')
    df[Config.min_col] = df[tgt_date_col].dt.round('15min')
    df[Config.min_col] = df[Config.min_col].dt.minute / 60
    df[Config.hr_col] = df[tgt_date_col].dt.hour
    df[Config.time_col] = df[Config.hr_col] + df[Config.min_col]
    df[Config.year_col] = df[tgt_date_col].dt.year
    df[Config.month_col] = df[tgt_date_col].dt.month
    # monday is 0, sunday is 6
    df[Config.day_col] = df[tgt_date_col].dt.day

    df = df.reset_index().rename(columns={'index': index_id})

    return df


def categorize_charge_cols(df):
    """
    assign category codes to special order
    """

    cols = df.columns.tolist()

    target_cols = [i for i in cols if "_class" in i]

    if "charges_class" in target_cols:
        target_cols.remove("charges_class")

    Config.my_logger.info(f'Categorizing {target_cols} in order of charge class severity.')

    df[target_cols] = df[target_cols].astype('object')
    # set the order of severity in charges from least to greatest
    Config.charge_order.reverse()
    Config.my_logger.info(f'Charge Severity Codes: {Config.charge_order}')
    # make target cols categorical
    for i in target_cols:
        df[i] = pd.Categorical(df[i], ordered=True, categories=Config.charge_order)
    # store new values as cat codes
    target_cols_cats = [f'{i}_cat_code' for i in target_cols]

    for idx, target_col in enumerate(target_cols):
        df[target_cols_cats[idx]] = df[target_col].cat.codes

    return df


def make_arrest_year_month(df, source_col='arrest_date', target_col1='arrest_year', target_col2='arrest_month'):
    df[target_col1]  = df[source_col].dt.year
    df[target_col2] = df[source_col].dt.month
    return df


def make_categorical(df, cols):
    df[cols] = df[cols].astype('category')
    return df


def parse_cols(df):
    Config.my_logger.info('Parsing column headers to lower case and replacing spaces with underscore.')
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    return df


def make_titlecase(df, cols):
    Config.my_logger.info(f'Converting data to titlecase and removing punctiontion for columns:\n{cols}')

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


def reduce_precision(df
                     , parse_col_names=True
                     , enable_mp=True
                     , mp_processors=None
                     , date_strings=None
                     , exclude_cols=None
                     , special_mappings=None
                     , bool_types=None
                     , categorical_ratio=.1
                     , categorical_threshold=20
                     , final_default_dtype='string'
                     ):
    """
    :params df: a pandas dataframe to optimize
    :params parse_col_names: Default to True; returns columns as lower case without spaces
    :params enable_mp: Default to True; runs optimization on columns in parallel. Set to false to run in series.
    :params date_strings: If None, default to a list of strings that indicate date columns -> ['_date', 'date_']
    :params exclude_cols: Default to None. A list of strings that indicate columns to exclude
    :params special_mappings: Default to None.
            A dictionary where each key is the desired dtype and the value are a list of strings that indicate columns
            to make that dtype.
    :params bool_types: If None, default to a list of values that indicate there is a boolean dtype such
            as True False, etc. -> [True, False, 'true', 'True', 'False', 'false']
    :params categorical_ratio: If None, default to .1 (10%). Evaluates the ratio of unique values in the column
            , if less than 10%, then, categorical.
    :params categorical_threshold: If None, default to 20. If the number of unique values is less than 20
            , make it a categorical column.
    :params final_default_dtype: If None, default to "string" dtype.
    """

    logging.info(f'Starting DataFrame Optimization. Starting with {mem_usage(df)} memory.')
    cols_to_convert = []
    # a default of strings that indicate the column is some kind of datetime column
    if date_strings is None:
        date_strings = ['_date', 'date_']
    # make a list of strings from all available dataframe columns
    cols_to_convert = [i for i in df.columns]
    # accommodate any special user-defined mappings
    special_exclusions = []
    if special_mappings is not None:
        for k, v in special_mappings.items():
            for i in v:
                df[i] = df[i].astype(k)
                special_exclusions.append(i)

    # exclude columns if a list is provided
    if exclude_cols is not None:
        cols_to_convert = [i for i in cols_to_convert if i not in exclude_cols]
    # by default, if special mappings are provided, exclude them from auto optimization
    if special_exclusions:
        cols_to_convert = [i for i in cols_to_convert if i not in special_exclusions]
    # by default, a list of values to be explicitly treated as bools
    if bool_types is None:
        bool_types = [True, False, 'true', 'True', 'False', 'false']
        # TODO: account for only T or only F or 1/0 situations
    if mp_processors is None:
        CPUs = mp.cpu_count() // 2
    # by default, enable multiprocessing to run optimizations in parallel
    if enable_mp:
        Config.my_logger.info('Starting optimization process with multiprocessor.')
        # break out the dataframe into a list of series to be worked on in parallel
        lst_of_series = [df[d] for d in cols_to_convert]

        pool = mp.Pool(CPUs)
        pbar = tqdm(lst_of_series, desc='Running DataFrame Optimization with multiprocessing')
        _reduce_precision_ = partial(_reduce_precision
                                     , date_strings=date_strings
                                     , bool_types=bool_types
                                     , categorical_ratio=categorical_ratio
                                     , categorical_threshold=categorical_threshold
                                     , final_default_dtype=final_default_dtype
                                     , enable_mp=enable_mp
                                     )
        list_of_converted = list(pool.imap(_reduce_precision_, pbar))
        pool.close()
        pool.join()

        # update the dataframe based on the converted records
        for (col_name, col_series) in list_of_converted:
            df[col_name] = col_series
    else:
        logging.info('Starting optimization process in series.')
        # un comment below to do conversion without MP
        df[cols_to_convert] = df[cols_to_convert].apply(lambda x: _reduce_precision(x
                                                                                    , date_strings=date_strings
                                                                                    , bool_types=bool_types
                                                                                    , categorical_ratio=categorical_ratio
                                                                                    , categorical_threshold=categorical_threshold
                                                                                    , final_default_dtype=final_default_dtype
                                                                                    , enable_mp=enable_mp
                                                                                    ))

    logging.info(f'Converted DF with new dtypes as follows:\n{df.dtypes}')
    logging.info(f'Completed DataFrame Optimization. Ending with {mem_usage(df)} memory.')

    return df


def _reduce_precision(x
                      , date_strings
                      , bool_types
                      , categorical_ratio
                      , categorical_threshold
                      , final_default_dtype
                      , enable_mp
                      ):
    """
    :params x: a pandas series (a column) to convert for dtype precision reduction
    """
    col_name = x.name
    col_type = x.dtype
    # return a unique list of non-na values in the current series
    unique_data = list(x.dropna().unique())

    n_unique = float(len(unique_data))
    n_records = float(len(x))
    cat_ratio = n_unique / n_records

    if 'int' in str(col_type):
        # if integer, make it the smallest possible type of integer
        c_min = x.min()
        c_max = x.max()

        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            x = x.astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            x = x.astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            x = x.astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            x = x.astype(np.int64)
            # TODO: set precision to unsigned integers with nullable NA

    elif 'float' in str(col_type):
        # if float, make it the smallest possible type of float
        c_min = x.min()
        c_max = x.max()
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                x = x.astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            x = x.astype(np.float32)
        else:
            x = x.astype(np.float64)

    elif 'datetime' in col_type.name or any(i in str(x.name).lower() for i in date_strings):
        # if datetime, make it datetime or if the col name matches default date strings
        try:
            x = pd.to_datetime(x)
        except:
            # TODO: conform to PEP and avoid naked except statement
            pass

    elif any(i in bool_types for i in unique_data):
        # make bool types as boolean instead of bool to allow for nullable bools
        x = x.astype('boolean')

    elif cat_ratio < categorical_ratio or n_unique < categorical_threshold:
        # if the category ratio is smaller than default thresholds, then make the column a categorical
        # a high level attempt to strike a balance when making columns categorical or not
        try:
            # return normal categories, i.e. avoid "dog" and "Dog" as different categories
            x = x.str.title()
        except:
            # TODO: conform to PEP and avoid naked except statement
            pass

        x = pd.Categorical(x)

    elif all(isinstance(i, str) for i in unique_data):
        # if all else fails, provide a final dtype as default
        x = x.astype(final_default_dtype)

    if enable_mp:
        return col_name, x
    else:
        return x


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
