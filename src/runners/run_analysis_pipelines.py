from src.stages.stage_analysis import stage_analysis_timeofday


def run_analysis_pipeline(df, data_folder, figures_folder):
    stage_analysis_timeofday(df=df, data_folder=data_folder, figures_folder=figures_folder)
