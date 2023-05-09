
from kedro.pipeline import Pipeline, node

from .nodes import extract_training_data,treat_missing_val,train_data_split,label_encoding,scaling_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func = extract_training_data,
                 inputs="weather_AUS_raw",
                 outputs="df1",
                 name = "extract_training_data_node",
            ),
            node(
                 func = treat_missing_val,
                 inputs="df1",
                 outputs="df1_treat_missing_value",
                 name = "treat_missing_val_node",
            ),
            node(
                 func = train_data_split,
                 inputs="df1_treat_missing_value",
                 outputs=["X_training","y_training"],
                 name = "train_data_split_node",
            ),
            node(
                 func = label_encoding,
                 inputs="X_training",
                 outputs="X_training0",
                 name = "label_encoding_node",
            ),
            node(
                 func = scaling_data,
                 inputs="X_training0",
                 outputs="df1_scaled_data1",
                 name = "scaling_data_node",
            ),
        
          ]
    )