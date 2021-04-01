# https://stackoverflow.com/questions/59390562/how-to-use-time-as-x-axis-for-a-scatterplot-with-seaborn

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.dates as mdates
from math import pi
import seaborn as sns
from src.utilities.config_general import *

plt.style.use('seaborn')


def time_of_day_analysis(df, figures_folder, target_col = 'charge_1_description_category_micro'):
    logging.info('Running time_of_day_analysis()')

    make_radar_fig(df, figures_folder)


def make_radar_fig(df, figures_folder):
    min_date = min(df['arrest_date']).year
    max_date = max(df['arrest_date']).year

    time_col = 'arrest_time'
    year_col = 'arrest_year'
    month_col = 'arrest_month'
    day_col = 'arrest_day'

    df[time_col] = df['arrest_date'].dt.hour
    df[year_col] = df['arrest_date'].dt.year
    df[month_col] = df['arrest_date'].dt.month
    # monday is 0, sunday is 6
    df[day_col] = df['arrest_date'].dt.dayofweek

    df = df.reset_index().rename(columns={'index': 'arrest_id'})

    df['lead_charge'] = df['charge_1_class']
    df['lead_charge_code'] = df['charge_1_class'].cat.codes

    counted = df.groupby(['lead_charge_code', year_col, month_col, day_col, time_col]).agg(
        arrest_count=('arrest_id', 'count')).reset_index()

    counted = counted[counted['lead_charge_code'] >= 0]

    counted["lead_charge_type"] = np.where(counted['lead_charge_code'] > 7, 'Felony', "None")
    counted["lead_charge_type"] = np.where((counted['lead_charge_code'] > 4) & (counted['lead_charge_code'] <= 7),
                                           'Misdemeanor', counted['lead_charge_type'])
    counted["lead_charge_type"] = np.where((counted['lead_charge_code'] > 0) & (counted['lead_charge_code'] <= 4),
                                           'Petty or Other', counted['lead_charge_type'])

    counted = counted.reset_index(drop=True)

    categories = counted[time_col].astype('str').unique().tolist()
    categories.sort()

    N = len(categories)

    values = [n for n in range(N)]
    angles = [n / float(N) * 2 * pi for n in range(N)]

    map_v_a = dict(zip(values, angles))

    counted['arrest_time_angle'] = counted[time_col].map(map_v_a)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar="True")

    ax.fill_between(counted['arrest_time_angle']
                    , counted['arrest_count']
                    , alpha=.2
                    , color='cornflowerblue'
                    )

    misdemeanors = counted[counted['lead_charge_type'] == 'Misdemeanor'].reset_index(drop=True)
    plt.polar(misdemeanors['arrest_time_angle']
              , misdemeanors['arrest_count']
              , linewidth=.1
              , alpha=.5
              , color="gray"
              )

    felonies = counted[counted['lead_charge_type'] == 'Felony'].reset_index(drop=True)
    plt.polar(felonies['arrest_time_angle']
              , felonies['arrest_count']
              , linewidth=.1
              , color="blue"
              )

    petty_other = counted[counted['lead_charge_type'] == 'Petty or Other'].reset_index(drop=True)
    plt.polar(petty_other['arrest_time_angle']
              , petty_other['arrest_count']
              , linewidth=.1
              , color="darkkhaki"
              )

    custom_lines = [Line2D([0], [0], color="darkkhaki", lw=4, alpha=.5),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="gray", lw=4),
                    Line2D([0], [0], color="cornflowerblue", lw=4),
                    ]

    plt.legend(custom_lines
               , ['Petty or Other', 'Felony', 'Misdemeanor', 'All']
               # , ncol=4
               # , fontsize='small'
               # , loc="lower center"
               , bbox_to_anchor=(0.1, 1)
               )

    ax.set_rlabel_position(0)
    plt.title(
        f'Chicago Police Department Arrest Analysis - Arrests by Time of Day (24 hr Clock)\nFrom {min_date} to {max_date}')
    plt.xticks(angles, values)
    plt.xlabel('Count of Arrests')
    plt.ylabel('Time of Day (24 hr)', labelpad=20)
    plt.tight_layout()
    file_path = os.sep.join([figures_folder, 'tod_1_all_radar.png'])
    plt.savefig(file_path)
    plt.show()


    ## petty or other only

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar="True")

    petty_other = counted[counted['lead_charge_type'] == 'Petty or Other'].reset_index(drop=True)

    ax.fill_between(petty_other['arrest_time_angle']
                    , petty_other['arrest_count']
                    , alpha=.2
                    , color='darkkhaki'
                    )

    plt.polar(petty_other['arrest_time_angle']
              , petty_other['arrest_count']
              , linewidth=.1
              , color="darkkhaki"
              )

    custom_lines = [Line2D([0], [0], color="darkkhaki", lw=4, alpha=.5),
                    # Line2D([0], [0], color="darkkhaki", lw=4),
                    ]

    plt.legend(custom_lines
               , ['Petty or Other']
               # , ncol=4
               # , fontsize='small'
               # , loc="lower center"
               , bbox_to_anchor=(0.1, 1)
               )

    ax.set_rlabel_position(0)
    plt.title(
        f'Chicago Police Department Arrest Analysis - Petty Arrests by Time of Day (24 hr Clock)\nFrom {min_date} to {max_date}')
    plt.xticks(angles, values)
    plt.xlabel('Count of Arrests')
    plt.ylabel('Time of Day (24 hr)', labelpad=20)
    plt.tight_layout()
    file_path = os.sep.join([figures_folder, 'tod_2_radar_petty.png'])
    plt.savefig(file_path)
    plt.show()








    # group = [time_col, target_col]
    # x1 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    # x1[time_col] = pd.to_datetime(x1[time_col], format='%H:%M:%S')
    # x1 = x1.set_index(time_col)
    # n = len(x1)
    # plt.figure()
    # ax = sns.scatterplot(x=x1.index
    #                      , y=x1['count']
    #                      , hue=x1[target_col]
    #                      , legend=True
    #                      )
    # ax.xaxis.set_major_locator(mdates.HourLocator())
    # ax.set_xlim(x1.index[0], x1.index[-1])
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # ax.tick_params(axis="x", rotation=45)
    # plt.title(
    #     f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nLead Charge Category (Micro) n={n} from {min_date} to {max_date}')
    # ax.legend(loc='upper left', fontsize='xx-small', ncol=2)
    # filename = f'cpd_tod_{target_col}.png'
    # data_file = os.sep.join([figures_folder, filename])
    # plt.savefig(data_file)
    # plt.show()
    #
    # df[time_col] = df['arrest_date'].dt.time
    # target_col = 'charge_1_description_category_macro'
    # group = [time_col, target_col]
    # x2 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    # x2[time_col] = pd.to_datetime(x2[time_col], format='%H:%M:%S')
    # x2 = x2.set_index(time_col)
    # n = len(x2)
    #
    # plt.figure()
    # ax = sns.scatterplot(x=x2.index
    #                      , y=x2['count']
    #                      , hue=x2[target_col]
    #                      , legend=True
    #                      )
    # ax.xaxis.set_major_locator(mdates.HourLocator())
    # ax.set_xlim(x2.index[0], x2.index[-1])
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # ax.tick_params(axis="x", rotation=45)
    # plt.title(
    #     f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nLead Charge Category (Macro) n={n} from {min_date} to {max_date}')
    # ax.legend(loc='upper left', fontsize='x-small')
    # filename = f'cpd_tod_{target_col}.png'
    # data_file = os.sep.join([figures_folder, filename])
    # plt.savefig(data_file)
    # plt.show()
    #
    # df[time_col] = df['arrest_date'].dt.time
    # target_col = 'charge_1_class'
    # group = [time_col, target_col]
    # x3 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    # x3[time_col] = pd.to_datetime(x3[time_col], format='%H:%M:%S')
    # x3 = x3.set_index(time_col)
    # n = len(x3)
    #
    # plt.figure()
    # ax = sns.scatterplot(x=x3.index
    #                      , y=x3['count']
    #                      , hue=x3[target_col]
    #                      , legend=True
    #                      )
    # ax.xaxis.set_major_locator(mdates.HourLocator())
    # ax.set_xlim(x2.index[0], x2.index[-1])
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # ax.tick_params(axis="x", rotation=45)
    # ax.legend(loc='upper left', fontsize='x-small', ncol=2)
    # plt.title(f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nArrest Charge Class n={n} from {min_date} to {max_date}')
    # filename = f'cpd_tod_{target_col}.png'
    # data_file = os.sep.join([figures_folder, filename])
    # plt.savefig(data_file)
    # plt.show()