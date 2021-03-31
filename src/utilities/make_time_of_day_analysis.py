# https://stackoverflow.com/questions/59390562/how-to-use-time-as-x-axis-for-a-scatterplot-with-seaborn

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from src.utilities.config_general import *

plt.style.use('seaborn')

def time_of_day_analysis(df, figures_folder, target_col = 'charge_1_description_category_micro'):
    logging.info('Running time_of_day_analysis()')

    min_date = min(df['arrest_date']).year
    max_date = max(df['arrest_date']).year
    time_col = 'arrest_time'
    df[time_col] = df['arrest_date'].dt.time

    group = [time_col, target_col]
    x1 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    x1[time_col] = pd.to_datetime(x1[time_col], format='%H:%M:%S')
    x1 = x1.set_index(time_col)
    n = len(x1)

    plt.figure()
    ax = sns.scatterplot(x=x1.index
                         , y=x1['count']
                         , hue=x1[target_col]
                         , legend=True
                         )
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.set_xlim(x1.index[0], x1.index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", rotation=45)
    plt.title(
        f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nLead Charge Category (Micro) n={n} from {min_date} to {max_date}')
    ax.legend(loc='upper left', fontsize='xx-small', ncol=2)
    filename = f'cpd_tod_{target_col}.png'
    data_file = os.sep.join([figures_folder, filename])
    plt.savefig(data_file)
    plt.show()

    df[time_col] = df['arrest_date'].dt.time
    target_col = 'charge_1_description_category_macro'
    group = [time_col, target_col]
    x2 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    x2[time_col] = pd.to_datetime(x2[time_col], format='%H:%M:%S')
    x2 = x2.set_index(time_col)
    n = len(x2)

    plt.figure()
    ax = sns.scatterplot(x=x2.index
                         , y=x2['count']
                         , hue=x2[target_col]
                         , legend=True
                         )
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.set_xlim(x2.index[0], x2.index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", rotation=45)
    plt.title(
        f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nLead Charge Category (Macro) n={n} from {min_date} to {max_date}')
    ax.legend(loc='upper left', fontsize='x-small')
    filename = f'cpd_tod_{target_col}.png'
    data_file = os.sep.join([figures_folder, filename])
    plt.savefig(data_file)
    plt.show()

    df[time_col] = df['arrest_date'].dt.time
    target_col = 'charge_1_class'
    group = [time_col, target_col]
    x3 = df.groupby(group)[[target_col]].agg('count').rename(columns={target_col: 'count'}).reset_index()
    x3[time_col] = pd.to_datetime(x3[time_col], format='%H:%M:%S')
    x3 = x3.set_index(time_col)
    n = len(x3)

    plt.figure()
    ax = sns.scatterplot(x=x3.index
                         , y=x3['count']
                         , hue=x3[target_col]
                         , legend=True
                         )
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.set_xlim(x2.index[0], x2.index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc='upper left', fontsize='x-small', ncol=2)
    plt.title(f'Preliminary Time of Day Analysis for Chicago Police Department Arrests\nArrest Charge Class n={n} from {min_date} to {max_date}')
    filename = f'cpd_tod_{target_col}.png'
    data_file = os.sep.join([figures_folder, filename])
    plt.savefig(data_file)
    plt.show()