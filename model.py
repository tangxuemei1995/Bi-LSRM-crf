'''
此代码用于构建模型
'''
import numpy as np
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield, tag2label
from utils import get_logger
from eval import conlleval
from build_fliter import test_filter_list


class BiLSTM_CRF(object):

    def __init__(self, args, embeddings,  tag2label, word2id, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = word2id
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config
        self.filter_num = args.cnn_filter
        self.filter_size = args.cnn_filter_size

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.cnn_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(
            tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(
            tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(
            tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(
            dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")

            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)


    def cnn_layer_op(self):
        hidden_size = self.word_embeddings.shape[-1].value
        
        conv_weights = tf.get_variable(
            "conv_weights", [self.filter_size, hidden_size, self.filter_num],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv_bias = tf.get_variable(
            "conv_bias", [self.filter_num], initializer=tf.zeros_initializer())
        conv = tf.nn.conv1d(self.word_embeddings,
                            conv_weights,
                            stride=1,
                            padding='SAME',
                            name='conv') #bug [40,12]vs[40,14]
        self.cnn_output_layer = tf.nn.relu(tf.nn.bias_add(conv, conv_bias), name='relu')
        
        
    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.cnn_output_layer,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
            # self.logits = tf.Print(self.logits, [tf.shape(self.logits)],summarize=429)

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            # self.labels_softmax_ = tf.Print(self.labels_softmax_, [self.labels_softmax_])
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(
                    learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(
                g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(
                grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev, args):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            self.run_epoches(sess, train, dev, self.tag2label, saver, args)

    def test(self, test, args):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('===========测试集结果===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test, args)
            self.evaluate(label_list, seq_len_list, test)
            with open(self.result_path + "predict.txt", 'w', encoding='utf8') as f:
                for lab, l in zip(label_list, seq_len_list):
                    labels = lab[:l]
                    for l in labels:
                        _, id2tag = tag2label()
                        f.write("{}\n".format(id2tag[l]))
                    f.write('\n')

    def demo_one(self, sess, sent):
        """
        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        
        return tag

    def run_epoches(self, sess, train, dev, tag2label, saver, args):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        best_f1 = 0  #用于记录训练过程中最好的f1值
        for epoch in range(self.epoch_num):
            num_batches = (len(train) + self.batch_size - 1) // self.batch_size

            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            batches = batch_yield(train, self.batch_size, self.vocab,
                                   self.tag2label, shuffle=self.shuffle)
                                   
            for step, (seqs, labels) in enumerate(batches):
                sys.stdout.write(
                    ' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                    
                step_num = epoch * num_batches + step + 1
                feed_dict, _ = self.get_feed_dict(
                    seqs, labels, self.lr, self.dropout_keep_prob)
                _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                             feed_dict=feed_dict)
                if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                    
                    self.logger.info(
                        '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                    loss_train, step_num))

                self.file_writer.add_summary(summary, step_num)

                # if step + 1 == num_batches:
                #     saver.save(sess, self.model_path, global_step=step_num)

            self.logger.info('-----------验证集测试结果------------')
            label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev, args)
            f1 = self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

            if f1 > best_f1:
                best_f1 = f1
                saver.save(sess, self.model_path, global_step=step_num)
            print("BET_F1: {}".format(best_f1))

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
         # labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        # print(seqs)
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list,
                 }
 
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev, args):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        count_batch = 0
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):

            label_list_, seq_len_list_ = self.predict_one_batch(
                    sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
            count_batch += 1
            
        return label_list, seq_len_list
        

    def viterbi_decode_change(self, score, transition_params, fliter):
        """Decode the highest scoring sequence of tags outside of TensorFlow.

        This should only be used at test time.

        Args:
          score: A [seq_len, num_tags] matrix of unary potentials.
          transition_params: A [num_tags, num_tags] matrix of binary potentials.

        Returns:
          viterbi: A [seq_len] list of integers containing the highest scoring tag
              indices.
          viterbi_score: A float containing the score for the Viterbi sequence.
        """
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]

        for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)
        backpointers_fliter = []
        for b in backpointers:
            backpointers_fliter.append(b * fliter)

        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score




    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(
            seqs, dropout=0.9)
        # feed_dict, seq_len_list = self.get_feed_dict(
            # seqs,  dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(
                    logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
                
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        # for label_, (sent, _, _, tag) in zip(label_list, data):
        for label_, (sent,tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            # if  len(label_) != len(sent):
            #     print(sent)
            # print(len(label_))
            # print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(
            self.result_path, 'result_metric_' + epoch_num)
        pre, recall, f1 = conlleval(
            model_predict, label_path, metric_path)
        print("pre {}".format(pre))
        print("recall {}".format(recall))
        print("f1 {}".format(f1))
        # for _ in conlleval(model_predict, label_path, metric_path):
        #     self.logger.info(_)
        return f1
