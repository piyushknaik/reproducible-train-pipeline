import os
import random
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_reproducibility(seed=42):
    """Configures environment for deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    return seed

def prepare_data(seed, test_size=0.2):
    """Loads, splits, and scales the digits dataset."""
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def create_datasets(X_train, y_train, X_test, y_test, batch_size=32, seed=42):
    """Converts numpy arrays to batched TensorFlow Datasets."""
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=seed)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    return train_ds, test_ds

def build_model(seed, learning_rate=0.001):
    """Defines and compiles the Keras Sequential model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(
            64, activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            10, activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_evaluate():
    """Main execution pipeline."""
    SEED = set_reproducibility(42)
    
    # Data pipeline
    X_train, X_test, y_train, y_test = prepare_data(SEED)
    train_ds, test_ds = create_datasets(X_train, y_train, X_test, y_test, seed=SEED)
    
    # Model pipeline
    model = build_model(SEED)
    model.fit(train_ds, epochs=10, verbose=0)
    
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Final test accuracy: {acc:.4f}")
    return model, acc

if __name__ == "__main__":
    train_and_evaluate()
