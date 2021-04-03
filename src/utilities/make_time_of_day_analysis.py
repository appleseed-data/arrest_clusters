# https://stackoverflow.com/questions/59390562/how-to-use-time-as-x-axis-for-a-scatterplot-with-seaborn

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.dates as mdates
import datetime
from math import pi
import seaborn as sns
from src.utilities.config_general import *
from src.utilities.config_dataprep import prep_time_of_day, prep_beats
from scipy import stats
plt.style.use('seaborn')


def time_of_day_analysis(df
                         , data_folder
                         , figures_folder
                         , target_charge_class='charge_1_class'
                         , target_charge_name='lead_charge'
                         , target_charge_cat_num='lead_charge_code'
                         ):

    logging.info('Running time_of_day_analysis()')
    # return min and max dates
    min_date = min(df['arrest_date']).year
    max_date = max(df['arrest_date']).year
    # return hours of the day as categories for plotting (24 hr clock)
    plot_categories = df[hr_col].astype('str').unique().tolist()
    plot_categories.sort()
    # return number of categories for plotting
    n_plot_categories = len(plot_categories)
    # prepare polar plot with category values and angles
    values_plot = [n for n in range(n_plot_categories)]
    angles_plot = [n / float(n_plot_categories) * 2 * pi for n in range(n_plot_categories)]
    # extract lead charge for each target record
    df[target_charge_name] = df[target_charge_class]
    # extract ordered num category for each target record
    df[target_charge_cat_num] = df[target_charge_class].cat.codes

    grouping = [target_charge_cat_num, year_col, month_col, day_col, hr_col]

    charge_types = ['Felony', 'Misdemeanor', 'Petty or Other', 'Not Specified']
    colors = ['gray', 'blue', 'darkkhaki', 'red']
    colors = dict(zip(charge_types, colors))

    plot_params = {'All': {'figure_name': 'tod_arrests_radar_all.png', 'title_nuance': 'All Arrests'}
                , 'Felony': {'figure_name': 'tod_arrests_radar_felony.png', 'title_nuance': 'Felony Arrests'}
                , 'Misdemeanor': {'figure_name': 'tod_arrests_radar_misdemeanor.png', 'title_nuance': 'Misdemeanor Arrests'}
                , 'Petty or Other': {'figure_name': 'tod_arrests_radar_petty_other.png', 'title_nuance': 'Petty or Other Arrests'}
                , 'Not Specified': {'figure_name': 'tod_arrests_radar_not_specified.png', 'title_nuance': 'Unspecified Arrests'}
                  }

    make_radar_fig(df=df
                   , figures_folder=figures_folder
                   , plot_params=plot_params
                   , max_date=max_date
                   , min_date=min_date
                   , charge_types=charge_types
                   , colors=colors
                   , grouping=grouping
                   , values_plot=values_plot
                   , angles_plot=angles_plot
                   , target_charge_cat_num=target_charge_cat_num
                   )

    make_unit_stats(df, charge_types=charge_types, figures_folder=figures_folder)
    make_unit_network(df, charge_types=charge_types, figures_folder=figures_folder)


def make_unit_network(df, charge_types, figures_folder, target_charge_type='charge_1_description_category_macro'):

    lead_charge_code = 'lead_charge_code'
    lead_charge_code_type = f'{lead_charge_code}_type'

    data = df[['district'
            , 'unit'
            , 'beat'
            , 'arrest_year'
            , 'arrest_month'
            , 'arrest_day'
            , 'arrest_time'
            , target_charge_type
            , lead_charge_code
               ]].copy(deep=True)


    for charge_type in charge_types:
        if charge_type == 'Felony':
            data[lead_charge_code_type] = np.where(data[lead_charge_code] > 7
                                              , charge_type
                                              , "None")
        elif charge_type == 'Misdemeanor':
            data[lead_charge_code_type] = np.where((data[lead_charge_code] > 4) & (data[lead_charge_code] <= 7)
                                              , charge_type
                                              , data[lead_charge_code_type])
        elif charge_type == 'Petty or Other':
            data[lead_charge_code_type] = np.where((data[lead_charge_code] > 0) & (data[lead_charge_code] <= 4)
                                              , charge_type
                                              , data[lead_charge_code_type])
        elif charge_type == 'Not Specified':
            data[lead_charge_code_type] = np.where((data[lead_charge_code] < 0)
                                              , 'Not Specified'
                                              , data[lead_charge_code_type])



def make_unit_stats(df, charge_types, figures_folder, target_charge_type='lead_charge_code_type'):

    data = df[['beat', 'unit', 'arrest_time', 'lead_charge_code']].copy(deep=True)

    for charge_type in charge_types:
        if charge_type == 'Felony':
            data[target_charge_type] = np.where(data['lead_charge_code'] > 7
                                              , charge_type
                                              , "None")
        elif charge_type == 'Misdemeanor':
            data[target_charge_type] = np.where((data['lead_charge_code'] > 4) & (data['lead_charge_code'] <= 7)
                                              , charge_type
                                              , data[target_charge_type])
        elif charge_type == 'Petty or Other':
            data[target_charge_type] = np.where((data['lead_charge_code'] > 0) & (data['lead_charge_code'] <= 4)
                                              , charge_type
                                              , data[target_charge_type])
        elif charge_type == 'Not Specified':
            data[target_charge_type] = np.where((data['lead_charge_code'] < 0)
                                              , 'Not Specified'
                                              , data[target_charge_type])

    data = data.groupby(target_charge_type)

    for i, group in data:

        plt.figure()
        sns.histplot(data=group
                     , x='arrest_time'
                     , kde=True
                     , hue='unit'
                     , stat='probability'
                     , legend=False
                     )
        plt.title(f'Distribution of arrests by time of day and unit.\nGrouped by {i} Arrests.')
        plt.tight_layout()
        file_name = f'tod_arrests_hist_{i}.png'
        file_path = os.sep.join([figures_folder, file_name])
        plt.savefig(file_path)
        plt.show()


def make_radar_fig(df
                   , figures_folder
                   , max_date
                   , min_date
                   , values_plot
                   , angles_plot
                   , target_charge_cat_num
                   , charge_types
                   , colors
                   , plot_params
                   , grouping=None
                   , agg_col='arrest_id'
                   , agg_type='count'
                   , angle_name='arrest_time_angle'
                   , filter_outliers=3
                   , title_base='Chicago Police Department Arrest Analysis - Arrests by Time of Day (24 hr Clock)'
                   ):

    agg_name = f'{agg_col}_{agg_type}'
    zscore_col = f'{agg_name}_zscore'

    df = df.groupby(grouping).agg({agg_col: agg_type}).reset_index()
    df = df.rename(columns={agg_col: agg_name})

    target_charge_type = f'{target_charge_cat_num}_type'

    for plot_param_type, plot_param_info in plot_params.items():
        figure_name = plot_param_info['figure_name']
        title_nuance = plot_param_info['title_nuance']

        if plot_param_type == 'All':
            charge_types = charge_types
        elif plot_param_type == 'Felony':
            charge_types = ['Felony']
        elif plot_param_type == 'Misdemeanor':
            charge_types = ['Misdemeanor']
        elif plot_param_type == 'Petty or Other':
            charge_types = ['Petty or Other']
        elif plot_param_type == 'Not Specified':
            charge_types = ['Not Specified']

        title = f'{title_base}\n {title_nuance} From {min_date} to {max_date} n={len(df)}'

        for charge_type in charge_types:
            if charge_type == 'Felony':
                df[target_charge_type] = np.where(df[target_charge_cat_num] > 7
                                                  , charge_type
                                                  , "None")
            elif charge_type == 'Misdemeanor':
                df[target_charge_type] = np.where((df[target_charge_cat_num] > 4) & (df[target_charge_cat_num] <= 7)
                                                  , charge_type
                                                  , df[target_charge_type])
            elif charge_type == 'Petty or Other':
                df[target_charge_type] = np.where((df[target_charge_cat_num] > 0) & (df[target_charge_cat_num] <= 4)
                                                  , charge_type
                                                  , df[target_charge_type])
            elif charge_type == 'Not Specified':
                df[target_charge_type] = np.where((df[target_charge_cat_num] < 0)
                                                  , 'Not Specified'
                                                  , df[target_charge_type])

        df = df.reset_index(drop=True)

        map_values2angles = dict(zip(values_plot, angles_plot))

        df[angle_name] = df[hr_col].map(map_values2angles)

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(polar="True")
        legend_entries = []

        if plot_param_type == 'All':

            ax.fill_between(df[angle_name]
                            , df[agg_name]
                            , alpha=.2
                            , color='cornflowerblue'
                            )

            legend_entry = ('All', Line2D([0], [0], color="cornflowerblue", lw=4))

            legend_entries.append(legend_entry)

        for charge_type in charge_types:
            df_plot = df[df[target_charge_type] == charge_type].reset_index(drop=True)

            if filter_outliers is not None:
                df_plot[zscore_col] = np.abs(stats.zscore(df_plot[agg_name]))
                df_plot = df_plot[df_plot[zscore_col] <= filter_outliers].reset_index(drop=True)

            if len(charge_types) == 1:
                ax.fill_between(df_plot[angle_name]
                                , df_plot[agg_name]
                                , alpha=.2
                                , color=colors[charge_type]
                                )

                legend_entry = ('All', Line2D([0], [0], color=colors[charge_type], lw=4, alpha=.2))

                legend_entries.append(legend_entry)

            plt.polar(df_plot[angle_name]
                      , df_plot[agg_name]
                      , linewidth=.1
                      , color=colors[charge_type]
                      )

            legend_entry = (charge_type, Line2D([0], [0], color=colors[charge_type], lw=4))
            legend_entries.append(legend_entry)

        legend_symbols = []
        legend_names = []

        for i in legend_entries:
            legend_names.append(i[0])
            legend_symbols.append(i[1])

        plt.legend(legend_symbols
                   , legend_names
                   , bbox_to_anchor=(0.1, 1)
                   )
        ax.set_rlabel_position(0)

        if filter_outliers is not None:
            title = title + ' *Outliers Removed'

        plt.title(title)
        plt.xticks(angles_plot, values_plot)
        plt.xlabel('Count of Arrests')
        plt.ylabel('Time of Day (24 hr)', labelpad=20)
        plt.tight_layout()
        file_path = os.sep.join([figures_folder, figure_name])
        plt.savefig(file_path)

        plt.show()