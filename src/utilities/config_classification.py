from src.utilities.config_dataprep import parse_cols
from src.utilities.config_general import *

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from fuzzypanda import matching

import joblib

import multiprocessing as mp
from tqdm import tqdm
from functools import partial

CPUs = mp.cpu_count()

micro_col = 'micro_category'
macro_col = 'macro_category'
police_col = 'police_related'

col_nums = ['1', '2', '3', '4']
charge_columns = [f'charge_{col}_description' for col in col_nums]
charge_columns_micro = [f'charge_{col}_description_category_micro' for col in col_nums]
charge_columns_macro = [f'charge_{col}_description_category_macro' for col in col_nums]
police_related_flags = [f'charge_{col}_description_police_related' for col in col_nums]


def prep_crosswalk(filename, sheet_name='CPD_crosswalk_final'):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    df = parse_cols(df)

    micro_charge_map = df[['description', micro_col]].copy(deep=True)
    micro_charge_map = micro_charge_map.drop_duplicates()

    macro_charge_map = df[[micro_col, macro_col]].copy(deep=True)
    macro_charge_map = macro_charge_map.drop_duplicates()

    police_related_map = df[['description', 'police_related']].copy(deep=True)
    police_related_map = police_related_map.drop_duplicates()

    return df, micro_charge_map, macro_charge_map, police_related_map


def apply_crosswalk_directmatch(df, micro_charge_map=None, macro_charge_map=None, police_related_map=None):
    logging.info('Mapping CPD Crosswalk to classifications where there is a direct match.')

    if micro_charge_map is not None:
        # read and set descriptions to category as dictionary maps
        micro_charge_map = micro_charge_map.set_index('description')
        micro_charge_map = micro_charge_map.to_dict()[micro_col]
    if macro_charge_map is not None:
        # read and set descriptions to category as dictionary maps
        macro_charge_map = macro_charge_map.set_index(micro_col)
        macro_charge_map = macro_charge_map.to_dict()[macro_col]
    if police_related_map is not None:
        # read and set descriptions to category as dictionary maps
        police_related_map = police_related_map.set_index('description')
        police_related_map = police_related_map.to_dict()[police_col]

    # map any charges that match crosswalk exactly

    for col in col_nums:
        df[f'charge_{col}_description_category_micro'] = df[f'charge_{col}_description'].map(micro_charge_map)

    for col in col_nums:
        df[f'charge_{col}_description_category_macro'] = df[f'charge_{col}_description_category_micro'].map(macro_charge_map)

    for col in col_nums:
        df[f'charge_{col}_description_police_related'] = df[f'charge_{col}_description'].map(police_related_map)

    return df


def apply_crosswalk_fuzzymatch(df, micro_charge_map, macro_charge_map, police_related_map, max_edit_distance=4):
    logging.info('Mapping CPD Crosswalk to classifications where there is a fuzzy match.')

    # return a dictionary version of the cpd crosswalk
    micro_charge_dict = micro_charge_map.set_index('description')
    micro_charge_dict = micro_charge_dict.to_dict()[micro_col]
    # return a dictionary version of the cpd crosswalk
    macro_charge_map = macro_charge_map.set_index(micro_col)
    macro_charge_map = macro_charge_map.to_dict()[macro_col]
    # return a dictionary version of the police related crosswalk
    police_related_dict = police_related_map.set_index('description')
    police_related_dict = police_related_dict.to_dict()[police_col]
    # initialize empty list to hold results of fuzzy match
    results = []
    # create a nested list to iterate through for charge description classifications
    target_columns = list(zip(charge_columns, tuple(zip(charge_columns_micro, charge_columns_macro))))

    # iterate through target cols with dataframe
    for charge_description, categories in target_columns:
        # run the fuzzy match script on pandas series
        result = run_fuzzy(df=df
                           , charge_description=charge_description
                           , categories=categories
                           , micro_charge_map=micro_charge_map
                           , max_edit_distance=max_edit_distance
                           , micro_charge_dict=micro_charge_dict
                           )
        # run_fuzzy is returning a list of one (name and series pair)
        results.append(result[0])

    # initialize empty list to hold results of fuzzy match
    results = []
    # create a nested list to iterate through for police flag classifications
    target_columns = list(zip(charge_columns, police_related_flags))

    for charge_description, police_related_col in target_columns:
        result = run_fuzzy(df=df
                           , charge_description=charge_description
                           , max_edit_distance=max_edit_distance
                           , micro_charge_dict=micro_charge_dict
                           , police_related_map=police_related_map
                           , police_related_dict=police_related_dict
                           , police_related_col=police_related_col
                           )
        results.append(result[0])

    # bring it all back together and update the main dataframe
    for col_name, col_series in results:
        df[col_name] = col_series

    for col in col_nums:
        start_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Mapping Macro Categories for charges -{col}- for {start_count}')
        df[f'charge_{col}_description_category_macro'] = df[f'charge_{col}_description_category_micro'].map(macro_charge_map)
        end_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Mapped {start_count - end_count}')

    df = df.drop(columns=['flag'])

    logging.info('Completed Fuzzy Matches.')

    return df


def run_fuzzy(df
              , charge_description
              , max_edit_distance
              , categories=None
              , micro_charge_map=None
              , micro_charge_dict=None
              , police_related_map=None
              , police_related_dict=None
              , police_related_col=None
              ):

    logging.info(f'- Trying Fuzzy Match for {charge_description}')

    results = []

    if police_related_map is not None:
        logging.info(f'- Starting Police Related Flag Matching for {charge_description}')
        # rename the police related map for convenience
        mapping_df = police_related_map
        # do mapping where there is a description but no category mapping
        df['flag'] = ~df[charge_description].isna() & df[police_related_col].isna()
        # get a count of records that are eligible to be mapped
        start_counts = df['flag'].value_counts()
        logging.info(f'-- Mapping Status: {start_counts[True]} remaining that need to be mapped for {police_related_col}')

        # run a fuzzy match on columns that need to be matched
        df[df['flag'] == True] = do_fuzzy_match(left_dataframe=df[df['flag'] == True].copy()
                                                , right_dataframe=mapping_df
                                                , left_cols=charge_description
                                                , target_cols=police_related_col
                                                , max_edit_distance=max_edit_distance
                                                , mapping_dict=police_related_dict
                                                )
        # get a new count after doing matches
        df['flag'] = ~df[charge_description].isna() & df[police_related_col].isna()
        end_counts = df['flag'].value_counts()
        logging.info(f'-- After fuzzy match, there are {end_counts[True]} unmapped records remaining.')

        result = tuple((police_related_col, df[police_related_col]))
        results.append(result)

        return results

    else:

        # run fuzzy matching for micro and macro charge descriptions
        for category in categories:
            logging.info(f'- Starting Charge Category Description Matching for {category}')
            # forcing logic to only map NLP model to micro categories, map macro cats based on micro in next part
            mapping_df = micro_charge_map if 'micro' in category else None
            # do mapping where there is a description but no category mapping
            df['flag'] = ~df[charge_description].isna() & df[category].isna()
            # get a count of records that are eligible to be mapped
            start_counts = df['flag'].value_counts()
            logging.info(f'-- Mapping Status: {start_counts[True]} remaining that need to be mapped for {category}')

            if mapping_df is not None:

                # do fuzzy mapping on eligible records
                df[df['flag'] == True] = do_fuzzy_match(left_dataframe=df[df['flag'] == True].copy()
                                                        , right_dataframe=mapping_df
                                                        , left_cols=charge_description
                                                        , target_cols=category
                                                        , max_edit_distance=max_edit_distance
                                                        , mapping_dict=micro_charge_dict
                                                        )
                # get a new count after doing matches
                df['flag'] = ~df[charge_description].isna() & df[category].isna()
                end_counts = df['flag'].value_counts()
                logging.info(f'-- After fuzzy match, there are {end_counts[True]} unmapped records remaining.')

                result = tuple((category, df[category]))
                results.append(result)

        return results


def do_fuzzy_match(left_dataframe
                   , right_dataframe
                   , left_cols
                   , target_cols
                   , max_edit_distance
                   , mapping_dict
                   , right_cols='description'
                   ):
    # perform fuzzy match on source (left) column and category (right) column
    # the match returns a value that represents the source column but that matches the category column exactly -> proxy
    # use the proxy to match to the dictionary as a direct match
    N = len(left_dataframe)
    logging.info(f'-- Trying to map {N} records with fuzzy match.')
    logging.info(f'--- match left on {left_cols} | match right on {right_cols}')

    matching.get_fuzzy_columns(left_dataframe=left_dataframe
                              , right_dataframe=right_dataframe
                              , left_cols=[left_cols]
                              , right_cols=[right_cols]
                              , max_edit_distance=max_edit_distance)

    fuzzy_col = f'fuzzy_{left_cols}'

    left_dataframe[target_cols] = left_dataframe[fuzzy_col].map(mapping_dict)

    left_dataframe = left_dataframe.drop(columns=[fuzzy_col])

    return left_dataframe


def apply_manual_match(df, criteria):
    logging.info('Mapping manual matches for CTA.')

    # create a nested list to iterate through
    target_columns = list(zip(charge_columns, tuple(zip(charge_columns_micro, charge_columns_macro))))

    for issue in criteria:
        for charge_description, categories in target_columns:
            # logging.info(f'- Trying Fuzzy Match for {charge_description}')
            for category in categories:
                # do mapping where there is a description but no category mapping
                df['flag'] = ~df[charge_description].isna() & df[category].isna() & df[charge_description].str.startswith(issue[0])
                start_counts = df['flag'].value_counts()
                logging.info(f'-- Manually applying mapping for: {start_counts[True]} in {category} starting with {issue[0]}')
                if 'micro' in category:
                    df[category] = np.where(df['flag']==True, issue[1][0], df[category])
                if 'macro' in category:
                    df[category] = np.where(df['flag'] == True, issue[1][1], df[category])

    df = df.drop(columns=['flag'])

    return df
