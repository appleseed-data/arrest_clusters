# https://stackoverflow.com/questions/59390562/how-to-use-time-as-x-axis-for-a-scatterplot-with-seaborn

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.dates as mdates
import datetime

import seaborn as sns
from src.utilities.config import Config
from scipy import stats

from zincbase import KB

plt.style.use('seaborn')


def make_unit_network(df, charge_types, figures_folder, target_charge_type='charge_1_description_category_macro'):
    lead_charge_code = 'lead_charge_code'
    lead_charge_code_type = f'{lead_charge_code}_type'

    kb = KB()
    kb.name = 'cpd'
    data = df[['lead_charge_police_related'
             , 'beat'
             , 'unit'
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

    keys = ['unit', 'beat']

    for key in keys:

        fill_val = '000' if key == 'unit' else '0000' if key == 'beat' else 'None'
        data[key] = data[key].fillna(fill_val).astype(str)
        key_data = data[key].unique().tolist()
        key_data.sort()
        for i in key_data:
            kb.store(f'isA({i},{key})')

    unit_nodes = data[['unit', 'beat']].drop_duplicates()
    unit_nodes = list(zip(unit_nodes['unit'], unit_nodes['beat']))

    for (unit, beat) in unit_nodes:
        kb.store(f'assignedTo({beat}, {unit})')

    #TODO

    # test queries to check kb store
    # results = list(kb.query(f'isA(Unit, unit)'))
    # print(results)

    # kb.plot(plot_title='Testing')

    # TODO analyze where police related is True or False


def make_unit_stats(df
                    , charge_types
                    , figures_folder
                    , target_charge_type='lead_charge_code_type'
                    , filter_outliers=3
                    ):
    Config.my_logger.info('make_unit_stats() Starting Stats Analysis by Unit')

    data = df[['lead_charge_police_related', 'beat', 'unit', 'arrest_time', 'lead_charge_code']].copy(deep=True)

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
        sns.histplot(data=group
                     , x='arrest_time'
                     , kde=True
                     , hue='unit'
                     , stat='density'
                     , legend=False
                     , common_norm=True
                     , element='step'
                     , lw=.1
                     , line_kws=dict(linewidth=.5)
                     , fill=False
                     , alpha=.2

                     )
        plt.title(f'Distribution of arrests by time of day and unit.\nGrouped by {i} Arrests.')
        plt.tight_layout()
        file_name = f'tod_arrests_hist_{i}.png'
        file_path = os.sep.join([figures_folder, file_name])
        plt.savefig(file_path)
        plt.show()

        key = 'unit'
        fill_val = '000' if key == 'unit' else '0000' if key == 'beat' else 'None'

        group[key] = group[key].fillna(fill_val).astype('str')
        hue_order = group[key].unique().tolist()
        hue_order.sort()

        plt.figure()

        sns.set_style()

        g = sns.FacetGrid(group
                        , col='lead_charge_police_related'
                        , col_order=[True, False]
                        , hue=key
                        , hue_order=hue_order
                        , height=4.5
                        , aspect=1
                        )

        g.map(sns.histplot
              , 'arrest_time'
              , kde=True
              , common_norm=True
              , stat='density'
              , fill=False
              , element='step'
              , lw=.3
              , line_kws=dict(linewidth=.5)
              , alpha=.6
              )

        g.fig.suptitle(f'Distribution of arrests by time of day and unit.\nGrouped by {i} and Police-Related Arrests.')
        plt.tight_layout()
        file_name = f'tod_arrests_hist_{i}_policerelated.png'
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

    Config.my_logger.info('make_radar_fig() Starting Time of Day Analysis')
    agg_name = f'{agg_col}_{agg_type}'
    zscore_col = f'{agg_name}_zscore'

    df = df.groupby(grouping).agg({agg_col: agg_type}).reset_index()
    df = df.rename(columns={agg_col: agg_name})
    N_records = df[agg_name].sum()

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

        if min_date == max_date:
            title = f'{title_base}\n {title_nuance} During {min_date} n={N_records}'
        else:
            title = f'{title_base}\n {title_nuance} From {min_date} to {max_date} n={N_records}'

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

        df[angle_name] = df[Config.hr_col].map(map_values2angles)

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(polar="True")
        legend_entries = []

        if plot_param_type == 'All':

            plt.polar(df[angle_name]
                      , df[agg_name]
                      , linewidth=.1
                      , alpha=0
                      , color='gray'
                      )

            police_related = df[df['lead_charge_police_related'] == True].reset_index(drop=True)

            ax.fill_between(police_related[angle_name]
                            , police_related[agg_name]
                            , alpha=.2
                            , color='aqua'
                            )

            legend_entry = ('Is police-related', Line2D([0], [0], color="aqua", lw=4))
            legend_entries.append(legend_entry)

            police_related = df[df['lead_charge_police_related'] == False].reset_index(drop=True)

            ax.fill_between(police_related[angle_name]
                            , police_related[agg_name]
                            , alpha=.5
                            , color='silver'
                            )

            legend_entry = ('Not police-related', Line2D([0], [0], color="silver", lw=4))
            legend_entries.append(legend_entry)

        if plot_param_type != 'All':

            for charge_type in charge_types:
                df_plot = df[df[target_charge_type] == charge_type].reset_index(drop=True)

                if filter_outliers is not None:
                    df_plot[zscore_col] = np.abs(stats.zscore(df_plot[agg_name]))
                    df_plot = df_plot[df_plot[zscore_col] <= filter_outliers].reset_index(drop=True)

                plt.polar(df_plot[angle_name]
                          , df_plot[agg_name]
                          , linewidth=.1
                          , alpha=.2
                          , color=colors[charge_type]
                          )

                legend_entry = (charge_type, Line2D([0], [0], color=colors[charge_type], lw=4))
                legend_entries.append(legend_entry)

                if len(charge_types) == 1:

                    police_related = df_plot[df_plot['lead_charge_police_related'] == False].reset_index(drop=True)

                    ax.fill_between(police_related[angle_name]
                                    , police_related[agg_name]
                                    , alpha=.5
                                    , color='silver'
                                    )

                    legend_entry = ('Not police-related', Line2D([0], [0], color="silver", lw=4))
                    legend_entries.append(legend_entry)

                    police_related = df_plot[df_plot['lead_charge_police_related'] == True].reset_index(drop=True)

                    ax.fill_between(police_related[angle_name]
                                    , police_related[agg_name]
                                    , alpha=.5
                                    , color='aqua'
                                    )

                    legend_entry = ('Is police-related', Line2D([0], [0], color="aqua", lw=4))
                    legend_entries.append(legend_entry)
                    n_records = df_plot[agg_name].sum()
                    pct_of_total = round((n_records / N_records) * 100, 1)
                    title = f'{title_base}\n {title_nuance} From {min_date} to {max_date} n={pct_of_total}% of Total'
                    if filter_outliers is not None:
                        title = title + ' *Outliers Removed'

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

        plt.title(title)
        plt.xticks(angles_plot, values_plot)
        plt.xlabel('Count of Arrests')
        plt.ylabel('Time of Day (24 hr)', labelpad=20)
        plt.tight_layout()
        file_path = os.sep.join([figures_folder, figure_name])
        plt.savefig(file_path)

        plt.show()
