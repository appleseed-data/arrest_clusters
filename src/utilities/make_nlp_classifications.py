from src.utilities.config_classification import *
import texthero as hero
from texthero import preprocessing as pp
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score


text_pipeline = [pp.lowercase
                   , pp.remove_diacritics
                   , pp.remove_digits
                   , pp.remove_stopwords
                   , pp.remove_punctuation
                   , pp.remove_whitespace
                   , pp.stem
                     ]


def make_nlp_classification_model_charge_descriptions(data_folder, df=None, filename='arrests_redacted.bz2'):

    if df is None:
        data_file = os.sep.join([data_folder, filename])
        logging.info(f'Starting NLP Pipeline from {data_file}')
        df = pd.read_pickle(data_file)

    known_classifications = df[['charge_1_description', 'charge_1_description_category_micro']].copy()
    known_classifications = known_classifications.dropna()
    known_classifications = known_classifications.reset_index(drop=True)
    known_classifications = known_classifications.rename(columns={'charge_1_description':'description_original'
                                                                  ,'charge_1_description_category_micro':'category'})

    known_classifications['description_cleaned'] = hero.clean(known_classifications['description_original'], pipeline=text_pipeline)

    x_train, x_test, y_train, y_test = tts(known_classifications[['description_cleaned']], known_classifications['category'], test_size=0.3, shuffle=True)

    nlp_ppl = Pipeline([
                        ('cv', CountVectorizer()),
                        ('clf', ComplementNB())
                        ])
    logging.info('Fit Train Predict Model')
    model = nlp_ppl.fit(x_train['description_cleaned'], y_train)
    y_pred = model.predict(x_test['description_cleaned'])
    y_true = y_test.tolist()
    acc = accuracy_score(y_true, y_pred)
    logging.info(f'Accuracy Score is {acc}')

    joblib.dump(model, model_save_path)
    logging.info(f'Saving Model to {model_save_path}')

    return model


def apply_nlp_classification_model_charge_descriptions(df
                                                       , model
                                                       , data_folder
                                                       , filename='CPD_crosswalk_final.xlsx'
                                                       , sheet_name='CPD_crosswalk_final'
                                                       ):
    # the charge description maps
    data_file = os.sep.join([data_folder, filename])
    crosswalk, micro_charge_map, macro_charge_map, police_related_map = prep_crosswalk(filename=data_file, sheet_name=sheet_name)

    logging.info('Applying NLP Model.')
    # create a nested list to iterate through
    target_columns = list(zip(charge_columns, tuple(zip(charge_columns_micro, charge_columns_macro))))

    # return a dictionary version of the cpd crosswalk
    macro_charge_map = macro_charge_map.set_index(micro_col)
    macro_charge_map = macro_charge_map.to_dict()[macro_col]

    # run nlp match in parallel
    pool = mp.Pool(CPUs)
    pbar = tqdm(target_columns, desc='Running DataFrame NLP match with multiprocessing')
    run_nlp_match_ = partial(run_nlp_match, df=df, model=model)
    results = list(pool.imap(run_nlp_match_, pbar))
    pool.close()
    pool.join()

    # uncomment next series to run nlp match in regular style
    # results = []
    # for target_column in target_columns:
    #     result = run_nlp_match(df, target_column, model)
    #     results.append(result)

    # bring results back together
    for col_name, col_series in results:
        df[col_name] = col_series

    for col in col_nums:
        start_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Mapping Macro Categories for charges -{col}- for {start_count}')
        df[f'charge_{col}_description_category_macro'] = df[f'charge_{col}_description_category_micro'].map(macro_charge_map)
        end_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Remaining NA count {end_count}')

    return df


def run_nlp_match(target_column, df, model):
    charge_description, categories = target_column
    logging.info(f'- Starting NLP match for {charge_description}')

    results = []

    logging.info(f'- Starting NLP match for {charge_description}')

    for category in categories:
        if 'micro' in category:
            # do mapping where there is a description but no category mapping
            df['flag'] = ~df[charge_description].isna() & df[category].isna()
            start_counts = df['flag'].value_counts()
            logging.info(f'-- In {category}, there are {start_counts[True]} to classify.')

            temp = hero.clean(df[df['flag'] == True][charge_description].copy(), pipeline=text_pipeline)
            idx = temp.index.values.tolist()
            predictions = model.predict(temp)

            df[category].update(pd.Series(predictions, name=charge_description, index=idx))

            df['flag'] = ~df[charge_description].isna() & df[category].isna()
            end_counts = df['flag'].value_counts()

            try:
                logging.info(f'The are {end_counts[True]} charges left to classify')
            except:
                logging.info(
                    f'Classification Status 100%. Started with {start_counts[True]} to map. Ended with {end_counts[False]} mapped. ')

            result = tuple((category, df[category]))

            results.append(result)

    # currently returning first element of a list of name and series pair
    return results[0]


def apply_nlp_match_police_related(df
                                   , data_folder
                                   , models_folder
                                   , model_file='arrest_police_flag_classification'
                                   , known_description='charge_1_description'
                                   , known_mapping='charge_1_description_police_related'
                                   ):

    model_path = os.sep.join([models_folder, model_file])

    if not os.path.exists(model_path):
        logging.info('apply_nlp_match_police_related() NLP model not found, learning a model for classification.')
        known_classifications = df[[known_description, known_mapping]].copy(deep=True)
        known_classifications = known_classifications.dropna()
        known_classifications = known_classifications.reset_index(drop=True)
        known_classifications = known_classifications.rename(columns={known_description: 'description_original', known_mapping: 'category'})

        known_classifications['category'] = known_classifications['category'].map({False:0
                                                                                   ,True:1})

        known_classifications['description_cleaned'] = hero.clean(known_classifications['description_original'], pipeline=text_pipeline)

        x_train, x_test, y_train, y_test = tts(known_classifications[['description_cleaned']],
                                               known_classifications['category'], test_size=0.3, shuffle=True)

        nlp_ppl = Pipeline([
            ('cv', CountVectorizer()),
            ('clf', ComplementNB())
        ])

        logging.info('Fit Train Predict Model')
        model = nlp_ppl.fit(x_train['description_cleaned'], y_train)
        y_pred = model.predict(x_test['description_cleaned'])
        y_true = y_test.tolist()
        acc = accuracy_score(y_true, y_pred)
        logging.info(f'Accuracy Score is {acc}')

        joblib.dump(model, model_path)
        logging.info(f'Saving Police Related Classification Model to {model_path}')

    else:
        logging.info(f'Found arrest classification model for police related flag at {model_path}')
        model = joblib.load(model_path)

    logging.info('Applying NLP Model.')
    # create a nested list to iterate through
    target_columns = list(zip(charge_columns, police_related_flags))
    # run nlp match
    for charge_description, category in target_columns:

        df['flag'] = ~df[charge_description].isna() & df[category].isna()
        start_counts = df['flag'].value_counts()
        logging.info(f'-- In {category}, there are {start_counts[True]} to classify.')

        temp = hero.clean(df[df['flag'] == True][charge_description].copy(), pipeline=text_pipeline)
        idx = temp.index.values.tolist()
        predictions = model.predict(temp)
        predictions = [False if i == 0 else True for i in predictions]

        df[category].update(pd.Series(predictions, name=charge_description, index=idx))

        df['flag'] = ~df[charge_description].isna() & df[category].isna()
        end_counts = df['flag'].value_counts()

        try:
            logging.info(f'The are {end_counts[True]} charges left to classify')
        except:
            logging.info(
                f'Classification Status 100%.')

    df = df.drop(columns=['flag'])

    return df