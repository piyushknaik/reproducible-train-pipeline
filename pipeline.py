import mlflow
import mlflow.tensorflow
from train import set_reproducibility, prepare_data, create_datasets, build_model


def run_mlflow_pipeline():
    # Set the experiment name
    mlflow.set_experiment("Digits_Classification")

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
        
        print(f"Run complete. Accuracy: {acc}")

if __name__ == "__main__":
    run_mlflow_pipeline()
