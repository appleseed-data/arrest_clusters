from src.utilities.make_classification import *
from src.utilities.config import Config
import texthero as hero
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import joblib
import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

def make_nlp_classification_model_charge_descriptions(data_folder
                                                      , models_folder
                                                      , model_name_charge_classification
                                                      , df=None
                                                      , filename='arrests_redacted.bz2'
                                                      , known_mapping='charge_1_description_category_micro'
                                                      ):

    model_path_charge_classification = os.sep.join([models_folder, model_name_charge_classification])

    if os.path.exists(model_path_charge_classification):
        Config.my_logger.info(f'Found existing model for charge description classification, loading it from {model_path_charge_classification}')
        model = joblib.load(model_path_charge_classification)
        return model
    else:
        Config.my_logger.info(f'Did not find NLP model for charge description classification at {model_path_charge_classification}, starting NLP model training pipeline.')
        if df is None:
            data_file = os.sep.join([data_folder, filename])
            Config.my_logger.info(f'Starting NLP Pipeline from {data_file}')
            df = pd.read_pickle(data_file)

        known_classifications = df[['charge_1_description', known_mapping]].copy()
        known_classifications = known_classifications.dropna()
        known_classifications = known_classifications.reset_index(drop=True)
        known_classifications = known_classifications.rename(columns={'charge_1_description': 'description_original'
                                                                      ,'charge_1_description_category_micro': 'category'})

        known_classifications['description_cleaned'] = hero.clean(known_classifications['description_original'], pipeline=Config.text_pipeline)

        x_train, x_test, y_train, y_test = tts(known_classifications[['description_cleaned']], known_classifications['category'], test_size=0.3, shuffle=True)

        Config.my_logger.info('Fit Train Predict Model')

        model = Config.nlp_ppl.fit(x_train['description_cleaned'], y_train)
        y_pred = model.predict(x_test['description_cleaned'])
        y_true = y_test.tolist()
        acc = accuracy_score(y_true, y_pred)
        Config.my_logger.info('==== Model Results')
        Config.my_logger.info(f'==== Accuracy Score is {acc}')

        labels = df[known_mapping].dropna().astype('str').unique().tolist()
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_df = pd.DataFrame(cm, columns=labels, index=labels)
        cm_matrix_data = os.sep.join([models_folder, 'arrest_charge_descr_confusion_matrix.csv'])
        cm_df.to_csv(cm_matrix_data)

        plt.figure(figsize=(15,15))
        cmap = plt.cm.get_cmap('viridis')

        plot_confusion_matrix(model
                              , x_test['description_cleaned']
                              , y_true
                              , display_labels=labels
                              # , normalize='all'
                              , include_values=False
                              , xticks_rotation=30
                              , cmap=cmap
                              )
        plt.yticks(fontsize="x-small")
        plt.xticks(fontsize="xx-small")

        cm_matrix_plot = os.sep.join([models_folder, 'arrest_charge_descr_confusion_matrix.png'])
        plt.title(f'Confusion Matrix for Arrest Charge Description Classification Model\nOverall Accuracy is {acc}. Train Size={len(x_train)} Test Size={len(x_test)}')
        plt.tight_layout()
        plt.savefig(cm_matrix_plot)
        plt.show()

        joblib.dump(model, model_path_charge_classification)
        Config.my_logger.info(f'Saving Model to {model_path_charge_classification}')

        return model


def apply_nlp_classification_model_charge_descriptions(df
                                                       , data_folder
                                                       , models_folder
                                                       , model_name_charge_classification
                                                       , filename='CPD_crosswalk_final.xlsx'
                                                       , sheet_name='CPD_crosswalk_final'
                                                       , enable_mp=True
                                                       ):

    model = make_nlp_classification_model_charge_descriptions(df=df
                                                              , data_folder=data_folder
                                                              , model_name_charge_classification=model_name_charge_classification
                                                              , models_folder=models_folder)
    # the charge description maps
    data_file = os.sep.join([data_folder, filename])
    crosswalk, micro_charge_map, macro_charge_map, police_related_map = prep_crosswalk(filename=data_file, sheet_name=sheet_name)

    Config.my_logger.info('Applying NLP Model.')
    # create a nested list to iterate through
    target_columns = list(zip(Config.charge_columns, tuple(zip(Config.charge_columns_micro, Config.charge_columns_macro))))

    # return a dictionary version of the cpd crosswalk
    macro_charge_map = macro_charge_map.set_index(Config.micro_col)
    macro_charge_map = macro_charge_map.to_dict()[Config.macro_col]

    if enable_mp:
        # run nlp match in parallel
        pool = mp.Pool(Config.CPUs)
        pbar = tqdm(target_columns, desc='Running DataFrame NLP match with multiprocessing')
        run_nlp_match_ = partial(run_nlp_match, df=df, model=model)
        results = list(pool.imap(run_nlp_match_, pbar))
        pool.close()
        pool.join()
    else:
        # run nlp match in series
        results = []
        for target_column in target_columns:
            result = run_nlp_match(df, target_column, model)
            results.append(result)

    # bring results back together
    for col_name, col_series in results:
        df[col_name] = col_series

    n_cols = len(Config.charge_columns)
    for idx in range(n_cols):
        col = Config.charge_columns_macro[idx]
        start_count = df[col].isna().sum()
        Config.my_logger.info(f'Mapping Macro Categories for charges -{col}- for {start_count}')
        df[col] = df[Config.charge_columns_micro[idx]].map(macro_charge_map)
        end_count = df[col].isna().sum()
        Config.my_logger.info(f'Remaining NA count {end_count}')

    return df


def run_nlp_match(target_column, df, model):
    charge_description, categories = target_column
    Config.my_logger.info(f'- Starting NLP match for {charge_description}')

    results = []

    Config.my_logger.info(f'- Starting NLP match for {charge_description}')

    for category in categories:
        if 'micro' in category:
            # do mapping where there is a description but no category mapping
            df['flag'] = ~df[charge_description].isna() & df[category].isna()
            start_counts = df['flag'].value_counts()
            Config.my_logger.info(f'-- In {category}, there are {start_counts[True]} to classify.')

            temp = hero.clean(df[df['flag'] == True][charge_description].copy(), pipeline=Config.text_pipeline)
            idx = temp.index.values.tolist()
            predictions = model.predict(temp)

            df[category].update(pd.Series(predictions, name=charge_description, index=idx))

            df['flag'] = ~df[charge_description].isna() & df[category].isna()
            end_counts = df['flag'].value_counts()

            try:
                Config.my_logger.info(f'The are {end_counts[True]} charges left to classify')
            except:
                Config.my_logger.info(
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
        Config.my_logger.info('apply_nlp_match_police_related() NLP model not found, learning a model for classification.')
        known_classifications = df[[known_description, known_mapping]].copy(deep=True)
        known_classifications = known_classifications.dropna()
        known_classifications = known_classifications.reset_index(drop=True)
        known_classifications = known_classifications.rename(columns={known_description: 'description_original', known_mapping: 'category'})

        known_classifications['category'] = known_classifications['category'].map({False: 0, True: 1})

        known_classifications['description_cleaned'] = hero.clean(known_classifications['description_original'], pipeline=Config.text_pipeline)

        x_train, x_test, y_train, y_test = tts(known_classifications[['description_cleaned']],
                                               known_classifications['category'], test_size=0.3, shuffle=True)

        Config.my_logger.info('Fit Train Predict Model')

        model = Config.nlp_ppl.fit(x_train['description_cleaned'], y_train)
        y_pred = model.predict(x_test['description_cleaned'])
        y_true = y_test.tolist()
        acc = accuracy_score(y_true, y_pred)
        Config.my_logger.info(f'Accuracy Score is {acc}')

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        Config.my_logger.info(f'CM tn {tn} ,fp {fp}, fn {fn}, tp {tp}')

        plt.figure()
        plot_confusion_matrix(model
                              , x_test['description_cleaned']
                              , y_true
                              )
        plt.title(f'Confusion Matrix for Police Related Flag Classification.\nOverall Accuracy Score is {acc}. Train Size={len(x_train)}. Test Size={len(x_test)}.')
        cm_matrix_plot = os.sep.join([models_folder, 'arrest_police_related_confusion_matrix.png'])
        plt.tight_layout()
        plt.savefig(cm_matrix_plot)
        plt.show()

        joblib.dump(model, model_path)
        Config.my_logger.info(f'Saving Police Related Classification Model to {model_path}')

    else:
        Config.my_logger.info(f'Found arrest classification model for police related flag at {model_path}')
        model = joblib.load(model_path)

    Config.my_logger.info('Applying NLP Model.')
    # create a nested list to iterate through
    target_columns = list(zip(Config.charge_columns, Config.police_related_flags))
    # run nlp match
    for charge_description, category in target_columns:

        df['flag'] = ~df[charge_description].isna() & df[category].isna()
        start_counts = df['flag'].value_counts()
        Config.my_logger.info(f'-- In {category}, there are {start_counts[True]} to classify.')

        temp = hero.clean(df[df['flag'] == True][charge_description].copy(), pipeline=Config.text_pipeline)
        idx = temp.index.values.tolist()
        predictions = model.predict(temp)
        predictions = [False if i == 0 else True for i in predictions]

        df[category].update(pd.Series(predictions, name=charge_description, index=idx))

        df['flag'] = ~df[charge_description].isna() & df[category].isna()
        end_counts = df['flag'].value_counts()

        try:
            Config.my_logger.info(f'The are {end_counts[True]} charges left to classify')
        except:
            Config.my_logger.info(
                f'Classification Status 100%.')

    df = df.drop(columns=['flag'])

    return df
