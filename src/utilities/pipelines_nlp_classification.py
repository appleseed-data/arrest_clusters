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

def make_nlp_classification_model(df=None):

    if df is None:
        filename = 'data/arrests_redacted_temp.bz2'
        logging.info(f'Starting NLP Pipeline from {filename}')
        df = pd.read_pickle('data/arrests_redacted_temp.bz2')

    text_pipeline = [pp.lowercase
                   , pp.remove_diacritics
                   , pp.remove_digits
                   , pp.remove_stopwords
                   , pp.remove_punctuation
                   , pp.remove_whitespace
                   , pp.stem
                     ]

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


def apply_nlp_classification_model(df, model):
    # the charge description maps
    crosswalk, micro_charge_map, macro_charge_map = prep_crosswalk(filename='data/CPD_crosswalk_final.xlsx',
                                                                   sheet_name='CPD_crosswalk_final')

    logging.info('Applying NLP Model.')
    col_nums = ['1', '2', '3', '4']

    charge_columns = [f'charge_{col}_description' for col in col_nums]
    charge_columns_micro = [f'charge_{col}_description_category_micro' for col in col_nums]
    charge_columns_macro = [f'charge_{col}_description_category_macro' for col in col_nums]
    # create a nested list to iterate through
    target_columns = list(zip(charge_columns, tuple(zip(charge_columns_micro, charge_columns_macro))))

    # return a dictionary version of the cpd crosswalk
    macro_charge_map = macro_charge_map.set_index(micro_col)
    macro_charge_map = macro_charge_map.to_dict()[macro_col]

    for charge_description, categories in target_columns:
        logging.info(f'- Starting NLP match for {charge_description}')

        for category in categories:
            if 'micro' in category:
                # do mapping where there is a description but no category mapping
                df['flag'] = ~df[charge_description].isna() & df[category].isna()
                start_counts = df['flag'].value_counts()
                logging.info(f'-- In {category}, there are {start_counts[True]} to classify.')

                temp = hero.clean(df[df['flag']==True][charge_description].copy(), pipeline=text_pipeline)
                idx = temp.index.values.tolist()
                predictions = model.predict(temp)

                df[category].update(pd.Series(predictions, name=charge_description, index=idx))

                df['flag'] = ~df[charge_description].isna() & df[category].isna()
                end_counts = df['flag'].value_counts()

                try:
                    logging.info(f'The are {end_counts[True]} charges left to classify')
                except:
                    logging.info(f'Classification Status 100%. Started with {start_counts[True]} to map. Ended with {end_counts[False]} mapped. ')

    for col in col_nums:
        start_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Mapping Macro Categories for charges -{col}- for {start_count}')
        df[f'charge_{col}_description_category_macro'] = df[f'charge_{col}_description_category_micro'].map(macro_charge_map)
        end_count = df[f'charge_{col}_description_category_macro'].isna().sum()
        logging.info(f'Remaining NA count {end_count}')


    df = df.drop(columns=['flag'])

    return df