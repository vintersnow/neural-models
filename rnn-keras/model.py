# import keras
from keras.models import Sequential
from keras.layers import Embedding, recurrent, Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from utils import Metrics


class Model(object):
    def __init__(self, args):
        self.args = args

        if args.model == 'rnn':
            cell_fn = recurrent.SimpleRNN
        elif args.model == 'gru':
            cell_fn = recurrent.GRU
        elif args.model == 'lstm':
            cell_fn = recurrent.LSTM
        # elif args.model == 'nas':
            # cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.model = model = Sequential()
        model.add(Embedding(args.vocab_size, args.rnn_size, mask_zero=True))

        model.add(cell_fn(args.rnn_size))
        # model.add(Dense(args.output_size, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      # loss='categorical_crossentropy',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train(self, train_x, train_y):
        args = self.args
        model = self.model

        (train_x, valid_x,
         train_y, valid_y) = train_test_split(train_x, train_y,
                                              test_size=0.1, random_state=42)

        print('valid_x', valid_x.shape, 'valid_y', valid_y.shape)

        metrics = Metrics()

        # one_hot_labels = to_categorical(train_y, num_classes=args.output_size)
        train_x = pad_sequences(train_x, padding='post', value=0)
        valid_x = pad_sequences(valid_x, padding='post', value=0)

        model.fit(train_x,
                  # one_hot_labels,
                  train_y,
                  epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  validation_data=[valid_x, valid_y],
                  callbacks=[metrics])

    def evaluate(self, test_x, test_y):
        args = self.args
        model = self.model

        # one_hot_labels = to_categorical(test_y, num_classes=args.output_size)
        x = pad_sequences(test_x, padding='post', value=0)

        score = model.evaluate(x,
                               # one_hot_labels,
                               test_y,
                               batch_size=args.batch_size)

        print('score: ', score)
