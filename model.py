import tensorflow as tf
import tensorflow.contrib


class LSTMNet(object):
    def __init__(self, vocabulary_size, embedding_size, step_num, hidden_units, batch_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.step_num = step_num
        self.hidden_units = hidden_units
        self.batch_size = batch_size

        self.input = tf.placeholder(tf.int32, shape=[self.batch_size, self.step_num])
        self.output = tf.placeholder(tf.int32, shape=[self.batch_size, self.step_num])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.batch_size])

        # 设置word embedding层
        with tf.name_scope("embedding_layer"):
            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            sentence = tf.nn.embedding_lookup(embeddings, self.input)

        # LSTM层
        with tf.name_scope("LSTM_scope"):
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence, sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

        # 全连接层
        with tf.name_scope('fully_connected_layer'):
            w_d = self.__weight_variable([self.hidden_units * 2, 10])
            b_d = self.__bias_variable([10])
            out = tf.reshape(output, [-1, self.hidden_units * 2])
            y = tf.add(tf.matmul(out, w_d), b_d)

        # 损失函数
        with tf.name_scope('loss'):
            self.logits = tf.reshape(y, [-1, self.step_num, 10])
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.output, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar('loss', self.loss)

    @staticmethod
    def __weight_variable(shape, stddev=0.1):
        initial = tf.random_normal(shape, stddev=stddev, name='W')
        return tf.Variable(initial)

    @staticmethod
    def __bias_variable(shape, stddev=1):
        initial = tf.random_normal(shape=shape, stddev=stddev, name='b')
        return tf.Variable(initial)
