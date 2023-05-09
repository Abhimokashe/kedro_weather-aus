from kedro.pipeline import Pipeline, node

from .nodes import train_test_splitting,logregAlgorithm,prediction,evaluation

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                 func = train_test_splitting,
                 inputs=["df1_scaled_data1","y_training"],
                 outputs=["X_train","X_test","y_train","y_test"],
                 name = "train_test_splitting_node",
            ),
            node(
                 func = logregAlgorithm,
                 inputs=["X_train","X_test","y_train","y_test"],
                 outputs= "model",
                 name = "logregAlgorithm_node",
            ),
            node(
                 func = prediction,
                 inputs = ["X_train","X_test","y_train","y_test","model"],
                 outputs = ["y_pred_train","y_pred_test"],
                 name = "prediction_node",
            ),
            node(
                 func = evaluation,
                 inputs=["y_pred_train","y_pred_test","y_train","y_test"],
                 outputs=["accuracy_score_train","accuracy_score_test"],
                 name = "evaluation_node",
            ),
            
        
          ]
    )