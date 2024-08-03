from typing import Dict, List, Tuple

import mlflow
from mlflow.entities import RunData


def fetch_logged_data(
    run_id: str,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str], List[str]]:
    """
    Fetches logged parameters, metrics, tags (excluding 'mlflow.' tags), and artifacts for a given MLflow run ID.

    Parameters:
    run_id (str): The unique identifier of the MLflow run.

    Returns:
    Tuple containing:
        - params (Dict[str, str]): Dictionary of parameters logged in the run.
        - metrics (Dict[str, float]): Dictionary of metrics logged in the run.
        - tags (Dict[str, str]): Dictionary of tags logged in the run, excluding 'mlflow.' tags.
        - artifacts (List[str]): List of artifact paths associated with the run.
    """
    client = mlflow.tracking.MlflowClient()
    run_data: RunData = client.get_run(run_id).data

    params: Dict[str, str] = run_data.params
    metrics: Dict[str, float] = run_data.metrics
    tags: Dict[str, str] = {
        k: v for k, v in run_data.tags.items() if not k.startswith("mlflow.")
    }

    artifacts: List[str] = [f.path for f in client.list_artifacts(run_id, "model")]

    return params, metrics, tags, artifacts
