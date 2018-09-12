from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback
import os
import time
import numpy as np


class TimeHistory(Callback):
    """Callback that saves times of each epoch"""

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class Estimator(object):
    """Estimator that allows training and evaluating the model
       with given dataset.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train(self, epochs=15, output_dir='./', sample_weight=False, early_stopping=False, patience=0, **kwargs):
        """Train model on training data. Wrapper for keras model train.

        # Arguments:
            epochs: Integer, number of epochs to train the model.
            output_dir: where to save logs.
            sample_weight: whether to use weighted loss function.
            early_stopping: whether to use early stopping.
            patience: when to early stop (after n epochs with no improvement).

        # Returns:
            dictionary of metrics
        """
        x_train, y_train = self.data.get_training_data()
        x_val, y_val = self.data.get_validation_data()

        sample_weight = compute_sample_weight('balanced', y_train) if sample_weight else None

        model_path = os.path.join(output_dir, 'best_model.tmp')

        time_callback = TimeHistory()

        callbacks = [
            time_callback,
            ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True),
            TensorBoard(log_dir=os.path.join(output_dir, 'logs'))
        ]

        if early_stopping:
            callbacks.append(EarlyStopping(patience=patience))

        history = self.model.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_val, y_val),
            sample_weight=sample_weight,
            callbacks=callbacks
        )

        self.model.model.load_weights(model_path)
        if os.path.exists(model_path):
            os.remove(model_path)

        return {
            "time_history": time_callback.times
        }

    def evaluate(self, data_x, data_y, **kwargs):
        """Evaluate model on given data.

        # Arguments:
            data_x: data to feed to classifier.
            data_y: gold labels

        # Returns:
            dictionary with metrics.
        """
        y_true = np.argmax(data_y, axis=1)
        y_pred = np.argmax(self.predict(data_x, **kwargs), axis=1)

        labels = self.data.label_encoder.inverse_transform(np.unique(y_true))

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred)
        avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        C = confusion_matrix(y_true, y_pred)

        return {
            "labels": labels.tolist(),
            "accuracy": acc,
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "avg_precision": avg_prec,
            "avg_recall": avg_rec,
            "avg_f1": avg_f1,
            "confusion_matrix": C.tolist()
        }

    def predict(self, data_x, **kwargs):
        """Generates predictions for input samples. Wrapper for keras model predict.

        # Arguments
            data_x: data to feed to classifier.
        """
        return self.model.model.predict(data_x)
