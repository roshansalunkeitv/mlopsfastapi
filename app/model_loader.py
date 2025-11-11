import mlflow
import mlflow.sklearn

def load_model(run_id=None):
    """
    Load a model from MLflow using the provided run_id.
    If no run_id is passed, loads a default model.
    """

    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        # Default run_id (replace with your actual run_id)
        model_uri = "runs:/83cdc2c12712403389f809ce869b64f8/model"

    model = mlflow.sklearn.load_model(model_uri)
    return model
