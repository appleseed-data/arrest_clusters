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
from scipy import stats
plt.style.use('seaborn')


def time_of_day_analysis(df, figures_folder, target_col = 'charge_1_description_category_micro'):
    logging.info('Running time_of_day_analysis()')

    make_radar_fig(df, figures_folder)


def make_radar_fig(df, figures_folder):
    min_date = min(df['arrest_date']).year
    max_date = max(df['arrest_date']).year

    min_col = 'arrest_minute'
    hr_col = 'arrest_hour'
    time_col = 'arrest_time'
    year_col = 'arrest_year'
    month_col = 'arrest_month'
    day_col = 'arrest_day'
    # ref: https://stackoverflow.com/questions/32344533/how-do-i-round-datetime-column-to-nearest-quarter-hour

    df[min_col] = df['arrest_date'].dt.round('15min')
    df[min_col] = df[min_col].dt.minute / 60
    df[hr_col] = df['arrest_date'].dt.hour
    df[time_col] = df[hr_col] + df[min_col]
    df[year_col] = df['arrest_date'].dt.year
    df[month_col] = df['arrest_date'].dt.month
    # monday is 0, sunday is 6
    df[day_col] = df['arrest_date'].dt.dayofweek

    df = df.reset_index().rename(columns={'index': 'arrest_id'})

    categories1 = df[hr_col].astype('str').unique().tolist()
    categories1.sort()

    N1 = len(categories1)

    values1 = [n for n in range(N1)]
    angles1 = [n / float(N1) * 2 * pi for n in range(N1)]

    df['lead_charge'] = df['charge_1_class']
    df['lead_charge_code'] = df['charge_1_class'].cat.codes

    counted = df.groupby(['lead_charge_code', year_col, month_col, day_col, hr_col]).agg(
        arrest_count=('arrest_id', 'count')).reset_index()

    counted["lead_charge_type"] = np.where(counted['lead_charge_code'] > 7, 'Felony', "None")
    counted["lead_charge_type"] = np.where((counted['lead_charge_code'] > 4) & (counted['lead_charge_code'] <= 7),
                                           'Misdemeanor', counted['lead_charge_type'])
    counted["lead_charge_type"] = np.where((counted['lead_charge_code'] > 0) & (counted['lead_charge_code'] <= 4),
                                           'Petty or Other', counted['lead_charge_type'])
    counted["lead_charge_type"] = np.where((counted['lead_charge_code'] < 0), 'Not Specified', counted['lead_charge_type'])

    counted = counted.reset_index(drop=True)

    categories2 = counted[hr_col].astype('str').unique().tolist()

    categories2.sort()

    N2 = len(categories2)

    values2 = [n for n in range(N2)]
    angles2 = [n / float(N1) * 2 * pi for n in range(N2)]

    map_v_a_2 = dict(zip(values2, angles2))

    counted['arrest_time_angle'] = counted[hr_col].map(map_v_a_2)
    # counted['arrest_time_angle'] = counted[hr_col].map(map_v_a_2)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar="True")

    ax.fill_between(counted['arrest_time_angle']
                    , counted['arrest_count']
                    , alpha=.2
                    , color='cornflowerblue'
                    )

    misdemeanors = counted[counted['lead_charge_type'] == 'Misdemeanor'].reset_index(drop=True)
    misdemeanors['arrest_count_zscore'] = np.abs(stats.zscore(misdemeanors['arrest_count']))
    misdemeanors = misdemeanors[misdemeanors['arrest_count_zscore'] <= 3].reset_index(drop=True)

    plt.polar(misdemeanors['arrest_time_angle']
              , misdemeanors['arrest_count']
              , linewidth=.1
              , alpha=.5
              , color="gray"
              )

    felonies = counted[counted['lead_charge_type'] == 'Felony'].reset_index(drop=True)
    felonies['arrest_count_zscore'] = np.abs(stats.zscore(felonies['arrest_count']))
    felonies = felonies[felonies['arrest_count_zscore'] <= 3].reset_index(drop=True)
    plt.polar(felonies['arrest_time_angle']
              , felonies['arrest_count']
              , linewidth=.1
              , color="blue"
              )

    petty_other = counted[counted['lead_charge_type'] == 'Petty or Other'].reset_index(drop=True)
    petty_other['arrest_count_zscore'] = np.abs(stats.zscore(petty_other['arrest_count']))
    petty_other = petty_other[petty_other['arrest_count_zscore'] <= 3].reset_index(drop=True)

    plt.polar(petty_other['arrest_time_angle']
              , petty_other['arrest_count']
              , linewidth=.1
              , color="darkkhaki"
              )

    not_specified = counted[counted['lead_charge_type'] == 'Not Specified'].reset_index(drop=True)
    not_specified['arrest_count_zscore'] = np.abs(stats.zscore(not_specified['arrest_count']))
    not_specified = not_specified[not_specified['arrest_count_zscore'] <= 3].reset_index(drop=True)

    plt.polar(not_specified['arrest_time_angle']
              , not_specified['arrest_count']
              , linewidth=.1
              , color="red"
              )

    custom_lines = [Line2D([0], [0], color="darkkhaki", lw=4, alpha=.5),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="gray", lw=4),
                    Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="cornflowerblue", lw=4),
                    ]

    plt.legend(custom_lines
               , ['Petty or Other', 'Felony', 'Misdemeanor', 'Not Specified', 'All']
               # , ncol=4
               # , fontsize='small'
               # , loc="lower center"
               , bbox_to_anchor=(0.1, 1)
               )

    ax.set_rlabel_position(0)
    plt.title(
        f'Chicago Police Department Arrest Analysis - Arrests by Time of Day (24 hr Clock)\nFrom {min_date} to {max_date} n={len(counted)} *Outliers Removed')
    plt.xticks(angles1, values1)
    plt.xlabel('Count of Arrests')
    plt.ylabel('Time of Day (24 hr)', labelpad=20)
    plt.tight_layout()
    file_path = os.sep.join([figures_folder, 'tod_1_all_radar_filter_outliers.png'])
    plt.savefig(file_path)
    plt.show()

    ## petty or other only

    ###

    ###

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar="True")

    petty_other = counted[counted['lead_charge_type'] == 'Petty or Other'].reset_index(drop=True)

    petty_other['arrest_count_zscore'] = np.abs(stats.zscore(petty_other['arrest_count']))

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
        f'Chicago Police Department Arrest Analysis - Petty Arrests by Time of Day (24 hr Clock)\nFrom {min_date} to {max_date} n={len(petty_other)}')
    plt.xticks(angles2, values2)
    plt.xlabel('Count of Arrests')
    plt.ylabel('Time of Day (24 hr)', labelpad=20)
    plt.tight_layout()
    file_path = os.sep.join([figures_folder, 'tod_2_radar_petty.png'])
    plt.savefig(file_path)
    plt.show()

    petty_other = petty_other[petty_other['arrest_count_zscore'] <= 3].reset_index(drop=True)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar="True")

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

    plt.polar(not_specified['arrest_time_angle']
              , not_specified['arrest_count']
              , linewidth=.1
              , color="red"
              )

    custom_lines = [Line2D([0], [0], color="darkkhaki", lw=4, alpha=.5),
                    Line2D([0], [0], color="red", lw=4),
                    ]

    plt.legend(custom_lines
               , ['Petty or Other', 'Not Specified']
               # , ncol=4
               # , fontsize='small'
               # , loc="lower center"
               , bbox_to_anchor=(0.1, 1)
               )

    ax.set_rlabel_position(0)
    plt.title(
        f'Chicago Police Department Arrest Analysis - Petty Arrests by Time of Day (24 hr Clock)\nFrom {min_date} to {max_date} n={len(petty_other)} *Outliers Removed')
    plt.xticks(angles2, values2)
    plt.xlabel('Count of Arrests')
    plt.ylabel('Time of Day (24 hr)', labelpad=20)
    plt.tight_layout()
    file_path = os.sep.join([figures_folder, 'tod_2_radar_petty_filter_outliers.png'])
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