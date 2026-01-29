import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from train import set_reproducibility, prepare_data, create_datasets, build_model


def run_mlflow_pipeline():
    # Set the experiment name
    mlflow.set_experiment("Digits_Classification")
    
    # Initialize MlflowClient for model tracking
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Digits_Classification")
    experiment_id = experiment.experiment_id

    with mlflow.start_run(run_name="Deterministic_Keras_Run"):
        
        SEED = 42
        set_reproducibility(SEED)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "seed": SEED,
            "batch_size": 32,
            "epochs": 10,
            "lr": 0.001
        })

        # Data
        X_train, X_test, y_train, y_test = prepare_data(SEED)
        train_ds, test_ds = create_datasets(X_train, y_train, X_test, y_test, seed=SEED)

        # Model & Training
        # Use autolog for automatic capture of metrics and model weights
        mlflow.tensorflow.autolog() 
        
        model = build_model(SEED)
        model.fit(train_ds, epochs=10, verbose=0)

        # Evaluation
        loss, acc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric("test_accuracy", acc)
        
        # Log and register the model using tensorflow
        mlflow.tensorflow.log_model(model, "model")
        
        # Register the model in MLflow Model Registry
        model_uri = f"runs://{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "Digits_Classifier")
        
        print(f"Run complete. Accuracy: {acc}")
        print(f"Model registered as 'Digits_Classifier' in MLflow Model Registry")
    
    # Track models: Search runs and collect artifacts
    print("\n--- Model Tracking ---")
    artifacts = []
    
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        order_by=["attributes.start_time DESC"]
    )

    for run in runs:
        run_id = run.info.run_id
        artifacts_from_run = client.list_artifacts(run_id)
        artifacts.append({
            'run_id': run_id,
            'artifacts': artifacts_from_run
        })

    print(f"Collected artifacts from {len(artifacts)} run(s)")
    
    # Search for all logged models (registered and unregistered)
    logged_models = client.search_logged_models(experiment_ids=[experiment_id])
    print(f"\nAll Models (Registered & Unregistered) in experiment: {len(logged_models)} model(s)")
    for model_info in logged_models:
        print(f"  - Model: {model_info.artifact_path}, Run ID: {model_info.run_id}")

if __name__ == "__main__":
    run_mlflow_pipeline()
