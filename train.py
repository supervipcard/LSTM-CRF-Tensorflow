import tensorflow as tf
from model import LSTMNet
from data_handing import json2dict
import numpy as np
import logging
import random


def training_batch(batch_size):
    data = json2dict(data_file)
    data = list(filter(lambda x: len(x[0]) <= step_num, data))
    random.shuffle(data)
    data_len = len(data)
    batch_num = int(data_len / batch_size)

    for batch in range(batch_num):
        length_list, sentences, signs = [], [], []
        for Id in range(batch * batch_size, (batch + 1) * batch_size):
            length_list.append(min(len(data[Id][0]), step_num))
            sentences.append(data[Id][0][0: step_num] + [padding] * (step_num - len(data[Id][0])))
            signs.append(data[Id][1][0: step_num] + [padding] * (step_num - len(data[Id][1])))
        yield np.array(length_list), np.array(sentences), np.array(signs)


# 训练
def train():
    logging.info('正在加载LSTM网络...')
    lstm = LSTMNet(vocabulary_size, embedding_size, step_num, hidden_units, batch_size)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('train'):
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('./model')
    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)

    logging.info('开始训练！')
    for epoch in range(epochs):
        length_list, sentence_list, sign_list = [], [], []
        for length, sentences, signs in training_batch(batch_size):
            length_list.append(length)
            sentence_list.append(sentences)
            sign_list.append(signs)

        for length, sentence, sign in zip(length_list, sentence_list, sign_list):
            feed_dict = {
                lstm.input: sentence,
                lstm.output: sign,
                lstm.sequence_lengths: length
            }
            _, step, rs, loss, lr = sess.run([train_op, global_step, merged, lstm.loss, learning_rate], feed_dict=feed_dict)
            writer.add_summary(rs, step)
            logging.info({'step': step, 'loss': loss, 'lr': lr})
        saver.save(sess, save_file, global_step=step)


# 测试
def test():
    logging.info('正在加载LSTM网络...')
    lstm = LSTMNet(vocabulary_size, embedding_size, step_num, hidden_units, 1)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('./model')
    if checkpoint:
        saver.restore(sess, checkpoint)

        words_dict = json2dict(vocabulary_file)

        sentence_1 = input('请输入句子：').replace(' ', '')
        sentence = [words_dict.get(word, 0) for word in sentence_1]
        sentence = np.array([sentence[0: step_num] + [padding] * (step_num - len(sentence))])

        feed_dict = {
            lstm.input: sentence,
            lstm.sequence_lengths: [len(sentence_1)]
        }
        logits, transition_params = sess.run([lstm.logits, lstm.transition_params], feed_dict=feed_dict)

        viterbi, _ = tf.contrib.crf.viterbi_decode(logits[0][0: len(sentence_1)], transition_params)
        print(viterbi)


if __name__ == '__main__':
    data_file = 'train_data.json'
    vocabulary_file = 'vocabulary.json'
    save_file = 'model/model.ckpt'
    padding = 0

    batch_size = 32
    vocabulary_size = len(json2dict(vocabulary_file))
    embedding_size = 100
    step_num = 500  # 句子中的最大字数
    hidden_units = 300   # LSTM cell的神经元个数

    initial_learning_rate = 0.01   # 学习速率
    decay_steps = 500
    decay_rate = 0.1
    epochs = 10

    max_grad_norm = 5

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    with tf.Session().as_default() as sess:
        train()
        # test()
