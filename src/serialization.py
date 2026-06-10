import os
import pickle
import xgboost as xgb


def serialize_model_artifacts(model, model_name):
    """Saves the trained model objects to the models/ directory for production reuse."""
    # THIS LINE FORCES WINDOWS/MAC TO CREATE THE 'models' FOLDER IF IT IS MISSING
    os.makedirs("models", exist_ok=True)

    if isinstance(model, xgb.XGBClassifier):
        file_path = os.path.join(
            "models", f"{model_name.lower().replace(' ', '_')}.json"
        )
        model.save_model(file_path)
    else:
        file_path = os.path.join(
            "models", f"{model_name.lower().replace(' ', '_')}.pkl"
        )
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    print(f"[Serialization] Saved model artifact successfully to: {file_path}")