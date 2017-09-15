# from sklearn.utils import shuffle
from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
# import numpy as np
# import tensorflow as tf
from keras.datasets import imdb
# from keras.preprocessing.sequence import pad_sequences
import argparse

from model import Model


def get_data(vocab_size):
    # padding, start token, oov
    num_words = vocab_size - 3
    return imdb.load_data(num_words=num_words,
                          seed=42,
                          start_char=1,
                          oov_char=2,
                          index_from=3)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--validate', type=bool, default=False,
                        help='validate run or full training. (Default: False)')
    parser.add_argument('--validate_size', type=int, default=100,
                        help='size of validation run. (Default: 100)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs.'
                        '(Default: logs)')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state. (Default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN. (Default: 2)')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas. (Default: lstm)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size. (Default: 50)')
    parser.add_argument('--seq_length', type=int, default=1000,
                        help='RNN sequence length. (Default: 1000)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs. (Default: 50)')
    # parser.add_argument('--save_every', type=int, default=1000,
    #                     help='save frequency')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='number of vocabulary. (Default: 10000)')
    parser.add_argument('--regular_weight', type=float, default=0.001,
                        help='weight of regularization. (Default: 0.001)')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value. (Default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate. (Default: 0.002)')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop. (Default: 0.97)')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights '
                        'in the hidden layer. (Default: 1.0)')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights '
                        'in the input layer. (Default: 1.0)')
    parser.add_argument('--train', type=bool, default=True,
                        help='train the model. (Default: True)')
    parser.add_argument('--test', type=bool, default=True,
                        help='test the test data. (Default: True)')
    # parser.add_argument('--init_from', type=str, default=None,
    #                     help="""continue training from saved model at this path. Path must contain files saved by previous training process:
    #                         'config.pkl'        : configuration;
    #                         'chars_vocab.pkl'   : vocabulary definitions;
    #                         'checkpoint'        : paths to model file(s) (created by tf).
    #                                               Note: this file contains absolute paths, be careful when moving files around;
    #                         'model.ckpt-*'      : file(s) with model definition (created by tf)
    #                        (Default: None)
    #                     """)

    args = parser.parse_args()
    args.output_size = 2
    return args


if __name__ == '__main__':
    args = get_args()

    (train_X, train_y), (test_X, test_y) = get_data(args.vocab_size)
    print('load data')

    model = Model(args)
    print('build model')

    if args.validate:
        size = args.validate_size
        train_X = train_X[:size]
        train_y = train_y[:size]
        test_X = test_X[:size]
        test_y = test_y[:size]

    if args.train:
        print('start training')
        model.train(train_X, train_y)
        print('end training')

    if args.test:
        pred_y = model.predict(test_X)
        true_y = test_y.tolist()
        # print('pred', pred_y, true_y)

        print('f1 score: ',
              f1_score(true_y, pred_y, average='macro'),
              'validation: %s' % str(args.validate))
