import keras
import numpy as np
from sklearn.metrics import f1_score as f1


class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        predict = (predict > 0.5).astype(int)
        self.f1s = f1(targ, predict, average='macro')
        print(' - f1: {0:.3f}'.format(self.f1s))
        return
