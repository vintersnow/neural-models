import tensorflow as tf
from tensorflow.contrib import rnn
# import numpy as np
# from tensorflow.contrib import legacy_seq2seq
import os
import time
import numpy as np
from utils import DataLoader
from keras.preprocessing.sequence import pad_sequences


class Model(object):
    def __init__(self, args, training=True):
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        output_kp = args.output_keep_prob
        input_kp = args.input_keep_prob
        # for _ in range(args.num_layers):
        for _ in range(1):
            cell = cell_fn(args.rnn_size)
            if training and (output_kp < 1.0 or
                             input_kp < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=input_kp,
                                          output_keep_prob=output_kp)
            cells.append(cell)

        # stacked_lstm = rnn.MultiRNNCell(cells, state_is_tuple=True)
        # self.stacked_lstm = stacked_lstm

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, None])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size])
        # self.initial_state = stacked_lstm.zero_state(args.batch_size,
        #                                              tf.float32)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable("output_w",
                                       [args.rnn_size, args.output_size])
            output_b = tf.get_variable("output_b", [args.output_size])

        embedding = tf.get_variable("embedding",
                                    [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        def rnn_loop(state, x):
            output, state = cell(x, state)
            outputs.append(output)
            return state

        state = self.initial_state
        with tf.variable_scope('RNN'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs,
                                               initial_state=state)

        output = outputs[:, -1, :]
        self.final_state = state
        # print('output', output.shape)

        self.logits = tf.matmul(output, output_w) + output_b

        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        # print('pred shape', self.pred.shape)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.targets,
            logits=self.logits,
            name='xentropy'
        )

        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        rnn_abs_weight = [tf.abs(v) for v in tvars
                          if v.name.startswith('RNN/')]
        weight_sum = sum(rnn_abs_weight)
        l1_loss = args.regular_weight * tf.reduce_sum(weight_sum)

        self.lr = tf.Variable(0.0, trainable=False)
        with tf.name_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)
            self.train_op = opt.minimize(self.cost + l1_loss)

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def train(self, train_X, train_y):
        args = self.args

        data_loader = DataLoader(args.batch_size, train_X, train_y)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            # instrument for tensorboard
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            # saver = tf.train.Saver(tf.global_variables())

            for e in range(args.num_epochs):
                sess.run(tf.assign(self.lr,
                                   args.learning_rate * (args.decay_rate ** e))
                         )

                data_loader.reset_batch_pointer()
                state = sess.run(self.initial_state)

                for b in range(data_loader.num_batches):

                    start = time.time()
                    x, y = data_loader.next_batch()

                    x = np.array(pad_sequences(x, padding='post', value=0))
                    y = np.array(y)
                    feed = {self.input_data: x, self.targets: y}
                    # print(s)
                    # feed[c] = initial_state[]
                    # for i, (c, h) in enumerate(self.initial_state):
                    #     feed[c] = state[i].c
                    #     feed[h] = state[i].h
                    train_loss, state, _ = sess.run([self.cost,
                                                     self.final_state,
                                                     self.train_op], feed)

                    # instrument for tensorboard
                    summ, train_loss, state, _ = sess.run([summaries,
                                                           self.cost,
                                                           self.final_state,
                                                           self.train_op],
                                                          feed)
                    writer.add_summary(summ, e * data_loader.num_batches + b)

                    end = time.time()
                    print('{}/{} (epoch {}), '
                          'train_loss = {:.3f}, '
                          'time/batch = {:.3f}'
                          .format(e * data_loader.num_batches + b,
                                  args.num_epochs * data_loader.num_batches,
                                  e, train_loss, end - start))

            model_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, model_path)
            print("model saved to {}".format(model_path))

    def predict(self, pred_X):
        args = self.args
        data_loader = DataLoader(args.batch_size, pred_X)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            model_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.restore(sess, model_path)
            # sess.run(tf.assign(self.lr,
            #                    args.learning_rate * (args.decay_rate ** e))
            #          )

            data_loader.reset_batch_pointer()
            # state = sess.run(self.initial_state)

            outputs = []
            for b in range(data_loader.num_batches):

                print('batch %d/%d' % (b, data_loader.num_batches))

                # start = time.time()
                x = data_loader.next_batch()[0]

                x = np.array(pad_sequences(x, padding='post', value=0))
                # y = np.array(y)
                feed = {self.input_data: x}
                # print(s)
                # feed[c] = initial_state[]
                # for i, (c, h) in enumerate(self.initial_state):
                #     feed[c] = state[i].c
                #     feed[h] = state[i].h
                pred = sess.run(self.pred, feed)

                outputs.extend(pred.flatten().tolist())

                # # instrument for tensorboard
                # summ, train_loss, state, _ = sess.run([summaries,
                #                                        self.cost,
                #                                        self.final_state,
                #                                        self.train_op],
                #                                       feed)
                # writer.add_summary(summ, e * data_loader.num_batches + b)

                # end = time.time()
                # print('{}/{} (epoch {}), '
                #       'train_loss = {:.3f}, '
                #       'time/batch = {:.3f}'
                #       .format(e * data_loader.num_batches + b,
                #               args.num_epochs * data_loader.num_batches,
                #               e, train_loss, end - start))
            return outputs
