import numpy as np


class DataLoader():
    def __init__(self, batch_size, *data):

        check_len = len(data[0])
        assert(all([len(x) == check_len for x in data]))

        self.len = check_len
        self.batch_size = batch_size
        self.data = data

        self.create_baatches()
        self.reset_batch_pointer()

    def create_baatches(self):
        self.num_batches = self.len // self.batch_size

        if self.num_batches == 0:
            assert False, ('Not enough data.'
                           'Make seq_length and batch_size small.')

        def make_batch(x):
            trimmed = x[:self.num_batches * self.batch_size]
            return np.split(trimmed, self.num_batches)

        self.batches = [make_batch(x) for x in self.data]

    def next_batch(self):
        batches = [x[self.pointer] for x in self.batches]
        self.pointer += 1
        return batches

    def reset_batch_pointer(self):
        self.pointer = 0
