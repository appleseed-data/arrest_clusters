from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from texthero import preprocessing as pp
import multiprocessing as mp
import time
import os
import logging

"""Runs system configurations
"""
from src.utilities import constants
from src.utilities.constants import *

import logging
import sys

import os
from os.path import join


def run_configuration():
    """Runs basic configuration for the workflow.
    """
    for folder in ALL_FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)

    logger = logging.getLogger("pipeline").getChild("configuration")
    format = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"

    time_string = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f'{time_string}_log.log'
    filename = join(constants.LOGGING_FOLDER, log_filename)

    log_formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    logging.basicConfig(filename=filename, format=format, level=logging.INFO)
    logging.getLogger().addHandler(stream_handler)

    logger.info("Logging configurations finished.")


class Config:
    """
    Config is a class of global variables to control aspects of the script.
    TODO: implement yaml files to control script run
    TODO: allow user-defined variables with init methods
    """
    CPUs = mp.cpu_count()

    redact_columns = ['first_name'
                    , 'last_name'
                    , 'middle_name'
                    , 'cb_no'
                    , 'case_number'
                    , 'street_no'
                    , 'street_dir'
                    , 'street_name'
                      ]

    drop_columns = ['charges_statute', 'charges_description', 'charges_type', 'charges_class', 'charges_fbi_code']

    micro_col = 'micro_category'
    macro_col = 'macro_category'
    police_col = 'police_related'

    min_col = 'arrest_minute'
    hr_col = 'arrest_hour'
    time_col = 'arrest_time'
    year_col = 'arrest_year'
    month_col = 'arrest_month'
    day_col = 'arrest_day'
    dtg_col = 'arrest_date'

    # this is the charge order from most severe to least severe
    charge_order = [
        'M', 'X', '1', '2', '3', '4'
        , 'A', 'B', 'C', 'L'
        , 'P', 'Z', 'U', "None"]

    n_charge_cols = 4
    charge_columns = [f'charge_{i}_description' for i in range(1, n_charge_cols+1)]

    charge_columns_micro = [f'{i}_category_micro' for i in charge_columns]
    charge_columns_macro = [f'{i}_category_macro' for i in charge_columns]
    police_related_flags = [f'{i}_police_related' for i in charge_columns]

    nlp_ppl = Pipeline([
        ('cv', CountVectorizer()),
        ('clf', ComplementNB())
    ])

    text_pipeline = [pp.lowercase
                   , pp.remove_diacritics
                   , pp.remove_digits
                   , pp.remove_stopwords
                   , pp.remove_punctuation
                   , pp.remove_whitespace
                   , pp.stem
                     ]




