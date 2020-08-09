'''
此代码为主函数，训练、测试、demo互相独立
'''

import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
import sys, pickle, os, random
from utils import str2bool, get_logger
from data import read_corpus, voc_build, tag2id, read_dictionary, random_embedding, load_bin_vec, get_W, one_hot
from metric import get_ner_demo
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 默认GPU1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 需要 ~700MB GPU 内存

np.random.seed(2019)
tf.set_random_seed(2019)

# 参数设置
parser = argparse.ArgumentParser(description='BiLSTM-CNN-CRF for Chinese NER task')
parser.add_argument('--all_data', type=str, default='data_path', help='all data source')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--dev_data', type=str, default='data_path', help='dev data source')
parser.add_argument('--batch_size', type=int, default=40, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=60, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--embeddingfile', type=str, default='emb', help=' char embedding file')
parser.add_argument('--embedding', type=str, default='cvzhihu.txt', help=' char embedding')
parser.add_argument('--shuffle', type=str2bool, default=False, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1575901637', help='model for test and demo')
parser.add_argument('--tags',type=str,default = 'O/PER-B/PER-I/PER-E/LOC-S/LOC-B/LOC-I/LOC-E/OFI-B/OFI-I/OFI-E/PER-S/OFI-S',help="tags for your data")
parser.add_argument('--cnn_filter',type=int,default=256,help="the number of cnn filter")
parser.add_argument('--cnn_filter_size',type=int,default=8,help="the size of cnn filter")
parser.add_argument('--model_name',type=str,default='model',help="the name of model")




class Train(object):
    '''
    训练
    '''
    def __init__(self, args, paths):
        self.paths = paths
        self.train_data, self.dev_data = self.get_data()
        self.word2id =  self.get_voc()
        self.embedding = self.get_embedding()
        self.tag2id,_= tag2id(args.tags)

    def get_data(self):
        '''
        读取训练数据和验证集数据
        :return:
        '''
        train_path = os.path.join('.', args.train_data, 'dev.txt')
        train_data = read_corpus(train_path)
        dev_path = os.path.join('.', args.dev_data, 'dev.txt')
        dev_data = read_corpus(dev_path)
        return train_data, dev_data

    def get_voc(self):
        '''
        训练时生成词表，生成词表时需要所有数据，all.txt
        :return:
        '''
        word2id_file ='./data_path/word2id.pkl'
        word2id = voc_build(os.path.join('.', args.train_data, 'all.txt'))
        with open(word2id_file, 'wb') as fw:
            pickle.dump(word2id, fw)
        return word2id
        print("word vocab size: {}".format(len(word2id)))

    def get_embedding(self):
        '''
        判断是预训练词向量还是随即初始化词向量
        :param args:
        :return:
        '''
        fname = os.path.join('.', args.embeddingfile, args.embedding)
        if args.pretrain_embedding == 'random':
            embeddings = random_embedding(self.word2id, args.embedding_dim)
        else:

            word_vecs = load_bin_vec(fname, vocab=list(self.word2id), ksize=300)
            embeddings = get_W(word_vecs=word_vecs, vocab_ids_map=self.word2id, k=300, is_rand=False)
        return embeddings


    def train_model(self):
        '''
        开始训练
        :return:
        '''
        model = BiLSTM_CRF(args, self.embedding, self.tag2id, self.word2id, self.paths, config=config)
        model.build_graph()
        print("train data: {}".format(len(self.train_data)))
        print("dev data: {}".format(len(self.dev_data)))
        model.train(self.train_data, self.dev_data, args)


class Test(object):
    '''
    测试
    '''
    def __init__(self, args, paths):
        self.paths = paths
        self.test_data = self.get_data()
        self.word2id = self.get_voc()
        self.embedding = self.get_embedding()
        self.tag2id,_ = tag2id(args.tags)

    def get_data(self):
        '''
        读取测试集
        :return:
        '''
        test_path = os.path.join('.', args.test_data, 'test.txt')
        test_data = read_corpus(test_path)
        return test_data

    def get_voc(self):
        '''
        读词表
        :return:
        '''
        word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
        print("word vocab size: {}".format(len(word2id)))
        return word2id


    def get_embedding(self):
        '''
        判断是预训练词向量还是随即初始化词向量
        :param args:
        :return:
        '''
        fname = os.path.join('.', args.embeddingfile, args.embedding)
        if args.pretrain_embedding == 'random':
            embeddings = random_embedding(self.word2id, args.embedding_dim)
        else:

            word_vecs = load_bin_vec(fname, vocab=list(self.word2id), ksize=300)
            embeddings = get_W(word_vecs=word_vecs, vocab_ids_map=self.word2id, k=300, is_rand=False)
        return embeddings

    def test_model(self,model_path):
        '''
        开始测试，测试时要传入模型地址
        :param model_path:
        :return:
        '''
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        self.paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, self.embedding, self.tag2id, self.word2id, self.paths, config=config)
        model.build_graph()
        print("test data size: {}".format(len(self.test_data)))
        model.test(self.test_data, args)



class Demo(object):
    '''
    demo
    '''
    def __init__(self, args, paths):
        self.paths = paths
        self.word2id = self.get_voc()
        self.embedding = self.get_embedding()
        self.tag2id, self.id2tag = tag2id(args.tags)

    def get_voc(self):
        '''
        读入词表
        :return:
        '''
        word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
        print("word vocab size: {}".format(len(word2id)))
        return word2id

    def get_embedding(self):
        '''
        判断是预训练词向量还是随即初始化词向量
        :param args:
        :return:
        '''
        fname = os.path.join('.', args.embeddingfile, args.embedding)
        if args.pretrain_embedding == 'random':
            embeddings = random_embedding(self.word2id, args.embedding_dim)
        else:

            word_vecs = load_bin_vec(fname, vocab=list(self.word2id), ksize=300)
            embeddings = get_W(word_vecs=word_vecs, vocab_ids_map=self.word2id, k=300, is_rand=False)
        return embeddings



    def demo_one(self,model_path):
        '''
        输入句子
        :param model_path:
        input:武三思與韋後日夜譖敬暉等不已
        :return: [[0, 2, 'PER'], [4, 5, 'PER'], [9, 10, 'PER']]
        '''
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        self.paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, self.embedding, self.tag2id, self.word2id, self.paths, config=config)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('begain to demo one sentence!')
            saver.restore(sess, ckpt_file)
            while (1):
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace() or demo_sent == 'end':
                    print('See you next time!')
                    break
                else:
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    print(get_ner_demo(tag))


if __name__=="__main__":

    args = parser.parse_args()
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join('.', args.train_data + "_save", args.model_name)
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))

    if args.mode == 'train':
        mode = Train(args,paths)
        mode.train_model()


    elif args.mode == 'test':
        mode = Test(args, paths)
        mode.test_model(model_path)

    elif args.mode == 'demo':
        mode = Demo(args, paths)
        mode.demo_one(model_path)
    else:
        print("please check the para --mode, choose train, test, demo!")