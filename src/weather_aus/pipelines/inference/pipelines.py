from kedro.pipeline import Pipeline, node

from .nodes import extracting_inference_data,splitting_inference_data,inference_data_treat_missing_val,inference_data_label_encoding,inference_data_scaling,logregAlgorithm1

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func = extracting_inference_data,
                 inputs="weather_AUS_raw",
                 outputs="df_inf",
                 name = "extracting_inference_data_node",
            ),
            node(
                 func = splitting_inference_data,
                 inputs="df_inf",
                 outputs=["X_inf","y_inf"],
                 name = "splitting_inference_data",
            ),
            node(
                 func = inference_data_treat_missing_val,
                 inputs="X_inf",
                 outputs="Xinf_treat_missing_value",
                 name = "inference_data_treat_missing_val_node",
            ),
            node(
                  func = inference_data_label_encoding,
                  inputs = "Xinf_treat_missing_value",
                  outputs = "Xinf_treat_missing_value0",
                  name = "inference_data_label_encoding_node",
            ),
            node(
                  func = inference_data_scaling,
                  inputs = "Xinf_treat_missing_value0",
                  outputs = "inference_scaled_data",
                  name = "inference_data_scaling_node",
            ),
            node(
                  func = logregAlgorithm1,
                  inputs = ["inference_scaled_data","model"],
                  outputs = "df1_y_pred_inf",
                  name = "logregAlgorithm1_node",
            )
            
        
          ]
    )