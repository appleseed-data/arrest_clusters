from src.stages.stage_analysis import stage_analysis_timeofday


def run_analysis_pipeline(df):
    stage_analysis_timeofday(df=df, target_year=None)
