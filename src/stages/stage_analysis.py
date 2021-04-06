from src.utilities.make_analysis_timeofday import make_radar_fig, make_unit_stats
from src.utilities.make_geospatial_analysis import geospatial_analysis
import logging

from math import pi

min_col = 'arrest_minute'
hr_col = 'arrest_hour'
time_col = 'arrest_time'
year_col = 'arrest_year'
month_col = 'arrest_month'
day_col = 'arrest_day'

def stage_analysis_timeofday(df
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
    # extract flag for whether lead charge is police related or not
    df['lead_charge_police_related'] = df['charge_1_description_police_related']

    grouping = ['lead_charge_police_related', target_charge_cat_num, year_col, month_col, day_col, hr_col]

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
