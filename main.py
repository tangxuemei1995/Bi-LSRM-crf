'''
此代码为主函数
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import os, argparse, time, random
from model import BiLSTM_CRF
import sys, pickle, os, random
from utils import str2bool, get_logger
from data import read_corpus, voc_build, tag2label, read_dictionary, random_embedding, load_bin_vec, get_W, one_hot
from sklearn.feature_extraction.text import CountVectorizer
from metric import  get_ner_demo
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  #默认GPU1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 需要 ~700MB GPU 内存

np.random.seed(2019)
tf.set_random_seed(2019)

#参数设置
parser = argparse.ArgumentParser(description='CNN-BiLSTM-CRF for Chinese NER task')
parser.add_argument('--all_data', type=str, default='data_path/guhanyu/data', help='all data source')
parser.add_argument('--train_data', type=str, default='data_path/guhanyu/data', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path/guhanyu/data', help='test data source')
parser.add_argument('--dev_data', type=str, default='data_path/guhanyu/data', help='dev data source')
parser.add_argument('--batch_size', type=int, default=60, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state for bi-lstm')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.8, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--embeddingfile', type=str, default='emb', help=' char embedding file')
parser.add_argument('--embedding', type=str, default='random', help=' char embedding')
parser.add_argument('--shuffle', type=str2bool, default=False, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1575901637', help='model for test and demo')
parser.add_argument('--cnn_filter', type=int, default=256, help='the number for cnn filter')
parser.add_argument('--cnn_filter_size', type=int, default=5, help='the size of cnn filter')
parser.add_argument('--model_path', type=str, default='model_1', help='the path for model save')

args = parser.parse_args()


def jiancibiao():
    '''
    此函数用于将所有统计所有字符，并存储
    建字表
    all_gu.txt 为所有语料的集合
    '''
    word2id_file ='./data_path/word2id.pkl'  #词表就放在固定的位置
    word2id, pos2id, d2id= voc_build(os.path.join('.', args.train_data, 'all_gu.txt'))
    with open(word2id_file, 'wb') as fw:
        pickle.dump(word2id, fw)
        
        
def get_embedding(word2id):
    '''
    读取预训练词向量或者随机生成词向量
    '''
    if args.pretrain_embedding == 'random':
         embeddings = random_embedding(word2id , args.embedding_dim)
    else:
        word_vecs = load_bin_vec(fname, vocab=list(word2id), ksize=300)
        embeddings = get_W(word_vecs=word_vecs, vocab_ids_map=word2id, k=300, is_rand=False)
    return embeddings

#训练模型
def train_model(embeddings,word2id, tag2id, paths):
    '''
    开始训练模型
    '''

    #读取训练集和验证集
    train_path = os.path.join('.', args.train_data, 'train_two_gu.txt')
    dev_path = os.path.join('.', args.dev_data, 'dev_two_gu.txt')
    train_data = read_corpus(train_path)
    train_size = len(train_data)
    dev_data = read_corpus(dev_path)
    dev_size = len(dev_data)
    print("训练集大小: {}".format(train_size))
    print("验证集大小: {}".format(dev_size))
    
    model = BiLSTM_CRF(args, embeddings, tag2id, word2id , paths, config=config)
    model.build_graph()
    model.train(train_data, dev_data, args) 

#测试模型
def test_model(embeddings, word2id, tag2id, paths):
    '''
    测试模型
    '''
    test_path = os.path.join('.', args.test_data, 'test_two_gu.txt')
    test_data = read_corpus(test_path)
    test_size = len(test_data)
    print("训练集大小: {}".format(test_size))
    
    ckpt_file = tf.train.latest_checkpoint(model_path) ##会自动找到最近保存的变量文件
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2id, word2id , paths, config=config)
    model.build_graph()
    model.test(test_data, args)
    
#预测一个句子
def get_entity(tag, demo_sent):
    '''从demo预测的结果中提取实体
    return:['[0,1]PER', '[7,8]LOC', '[10,11]PER']
    '''
    sent_result = []
    for i in range(len(tag)):
        sent_result.append([tag[i],demo_sent[i]])
    pred_all = []
    pred_one = []
    for tag_, char in sent_result:
        if tag_ == 0:
            tag_ = 'O'
        else:
            t_, head_ = tag_.split('-', 1)
            if head_ == 'M':
                head_ = 'I'
            tag_ = head_ + '-' + t_
        pred_one.append(tag_) 
    entity = get_ner_demo(pred_one)
    return entity
        
def demo_one(embeddings, word2id, tag2id,paths):
    '''
    输入一个句子进行识别
    '''
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2id, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('------------------demo------------------')
        saver.restore(sess, ckpt_file)
        while(1):
            print('请输入您的句子:')
            demo_sent = input()
            if demo_sent == '' or demo_sent == 'end' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sen = list(demo_sent.strip())
                demo_data = [(demo_sen, ['O'] * len(demo_sen))]
                tag = model.demo_one(sess, demo_data)
                entitys = get_entity(tag, demo_sen)   
                print('您输入的句子：', demo_sent)  
                print('实体标记：', entitys)  
                       
    
if __name__ == '__main__':
   
    #路径设置
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join('.', "data_path_save", args.model_path)
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
        # jiancibiao()
        pass
         #训练时先建词表,只有在训练时才新建词表，测试和训练要用同一个词表
        
    word2id = read_dictionary(os.path.join('.', 'data_path/' 'word2id.pkl'))  #读入词表
    print("词表长度: {}".format(len(word2id)))
    embeddings = get_embedding(word2id)
    tag2id, _ = tag2label()
 
    if args.mode == 'train':
        train_model(embeddings, word2id, tag2id, paths)
    elif args.mode == 'test':
        test_model(embeddings, word2id, tag2id, paths)
    elif args.mode == 'demo':
        demo_one(embeddings, word2id, tag2id, paths)
    
    
    
    