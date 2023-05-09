"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from weather_aus.pipelines import data_processing as dp
from weather_aus.pipelines import data_science as ds
from weather_aus.pipelines import inference as inf

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    inference_pipeline = inf.create_pipeline()
    
    return {
         "__default__": data_processing_pipeline + data_science_pipeline + inference_pipeline,
         "dp": data_processing_pipeline,
         "ds": data_science_pipeline,
         "inf": inference_pipeline
    }
