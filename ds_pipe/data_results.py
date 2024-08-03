"""
Cuarto step en el pipeline de DS.

El step final del pipeline es Feature Engineering.
Se encarga de la creación de nuevas variables a partir de las existentes.
Almacena las features en el directorio de features.
"""

import os
import pickle
from typing import Union

import numpy as np
import pandas as pd

from tools.dates import extract_date_features
from tools.eda import change_cols_order
from tools.sklearn_custom_estimators import CustomOneHotEncoder, CustomOrdinalEncoder
from utils.config import FEATURES_PATH, PARQUET_PATH


def main_results(parquet_name: Union[str, None] = None) -> None:
    """
    Process the data and generate various features and files.
    Steps:
        - .1. First Adjustments
        - .2. Updating Data Types
        - .3. Feature 'score' + 'ass_name_label'
        - .4. Feature 'submission_type'
        - .5. Features de tiempo
        - .6. Ordinal (Grouped) Features: 'particion' y 'activity_unlock_delta'
        - .7. Vector Target (Grouped)
        - .8. Grouped Feature 'score' + 'ass_name_label'
        - .9. Grouped Feature 'submission_type'
        - .10. Ensamble features and download
        - .11. Ensamble grouped features and download
        - .12. Descarga de los encoders

    Args:
        parquet_name (Union[str, None]): The name of the Parquet file to be processed.

    Raises:
        ValueError: If no file name is provided.

    Returns:
        None
    """
    print("  Step 4: Data Results Started  ".center(88, "."), end="\n\n")

    # .1. First Adjustments
    # Handling edge cases
    # Read Local Parquet file
    # Setting Environment variables
    if parquet_name is None:
        raise ValueError("No file name provided")
    if not parquet_name.endswith(".parquet"):
        parquet_name += ".parquet"
    cols = ["user_uuid", "course_uuid"]
    base_name = "_".join(parquet_name.split("_")[:-1])
    FEATURES_DIR = os.path.join(FEATURES_PATH, base_name)
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR, exist_ok=True)

    # .2. Updating Data Types
    # Encoding UUIDs to Ordinal
    # Convert TimeDelta to int
    df = pd.read_parquet(os.path.join(PARQUET_PATH, parquet_name))
    encoder_first_ordinal = CustomOrdinalEncoder(cols)
    df[cols] = encoder_first_ordinal.fit(df).transform(df).astype(int)
    df["activity_unlock_delta"] = df.activity_unlock_delta.dt.days

    # .3. Feature 'score' + 'ass_name_label'
    encoder_ass_name_label = CustomOneHotEncoder(["ass_name_label"])
    encoder_ass_name_label.fit(df).transform(df)
    scores_one_hot = (
        encoder_ass_name_label.X_encoded.values * df.score.values[:, np.newaxis]
    )
    feature_scores_one_hot = pd.DataFrame(
        scores_one_hot, columns=encoder_ass_name_label.X_encoded.columns
    )

    # .4. Feature 'submission_type'
    encoder_submission_type = CustomOneHotEncoder(["submission_type"])
    encoder_submission_type.fit(df).transform(df)
    feature_submission_one_hot = encoder_submission_type.X_encoded

    # .5. Features de tiempo
    feature_datetime = extract_date_features(
        df, "activity_date", include_col_name=False
    )

    # .6. Ordinal (Grouped) Features: 'particion' y 'activity_unlock_delta'
    grouped_data = df.groupby(cols)[["particion", "activity_unlock_delta"]].sum()
    encoder_ordinal = CustomOrdinalEncoder(["particion", "activity_unlock_delta"])
    feature_ordinal_grouped = (
        encoder_ordinal.fit(grouped_data).transform(grouped_data).astype(int)
    )

    # .7. Vector Target (Grouped)
    y = df.groupby(cols)["nota_final_materia"].mean().to_frame().reset_index(drop=False)

    # .8. Grouped Feature 'score' + 'ass_name_label'
    grouped_data = pd.concat([df[cols], feature_scores_one_hot], axis=1)
    feature_score_grouped = grouped_data.groupby(cols).sum().reset_index(drop=True)

    # .9. Grouped Feature 'submission_type'
    # aumentar el peso de los 'nombre_exam' x2
    feature_submission_one_hot["submission_type_nombre_examen"] = (
        feature_submission_one_hot["submission_type_nombre_examen"]
        * (nombre_examen_weight := 2)
    )
    grouped_data = pd.concat([df[cols], feature_submission_one_hot], axis=1)
    feature_submission_grouped = (
        grouped_data.groupby(cols).sum().astype(int).reset_index(drop=True)
    )

    # .10. Ensamble features and download
    file_name = f"{base_name}_features.parquet"
    ensamble_features = pd.concat(
        [df, feature_scores_one_hot, feature_submission_one_hot, feature_datetime],
        axis=1,
    )
    selected_init_cols = []
    selected_final_cols = ["nota_final_materia"]

    ensamble_features = change_cols_order(
        ensamble_features, selected_init_cols, selected_final_cols
    )
    ensamble_features = ensamble_features.rename(
        columns={"nota_final_materia": "target"}
    )
    ensamble_features = ensamble_features.sort_values(cols)
    ensamble_features = ensamble_features.reset_index(drop=True)
    ensamble_features.to_parquet(os.path.join(FEATURES_DIR, file_name))
    print(f"File saved as {file_name}.parquet in {FEATURES_DIR}", end="\n\n")

    # .11. Ensamble grouped features and download
    file_name = f"{base_name}_grouped_features.parquet"
    grouped_features = pd.concat(
        [feature_ordinal_grouped, feature_score_grouped, feature_submission_grouped, y],
        axis=1,
    )
    selected_init_cols = cols
    selected_final_cols = ["nota_final_materia"]
    order = (
        selected_init_cols
        + [
            col
            for col in grouped_features.columns
            if col not in selected_init_cols + selected_final_cols
        ]
        + selected_final_cols
    )
    grouped_features = grouped_features.reindex(columns=order)
    grouped_features = grouped_features.sort_values(cols)
    grouped_features.to_parquet(os.path.join(FEATURES_DIR, file_name))
    print(f"File saved as {file_name}.parquet in {FEATURES_DIR}", end="\n\n")

    # .12. Descarga de los encoders
    enc_label = [
        "encoder_first_ordinal",
        "encoder_ass_name_label",
        "encoder_submission_type",
        "encoder_ordinal",
    ]
    enc_obj = [
        encoder_first_ordinal,
        encoder_ass_name_label,
        encoder_submission_type,
        encoder_ordinal,
    ]
    encoders = dict(zip(enc_label, enc_obj))

    for label, obj in encoders.items():
        file_name = f"{base_name}_{label}.pkl"
        with open(os.path.join(FEATURES_DIR, file_name), "wb") as fp:
            pickle.dump(obj, fp)
        print(f"File saved as {file_name}.parquet in {FEATURES_DIR}", end="\n\n")

    print("  Step 4: Data Results Completed  ".center(88, "."), end="\n\n")
