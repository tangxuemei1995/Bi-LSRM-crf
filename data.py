import sys, pickle, os, random
import numpy as np
import os
import gensim, logging
import datetime
import gensim.models.keyedvectors as word2vec
import pandas
# from langconv import *

# tags, BME/S/O
def tag2id(tags):
    '''
    用于将tag 和 ID 组成字典
    
    '''
    tag2id,id2tag = {}, {}
    if tags == None:
        tag2id = {"O": 0, "PER-B": 1,
             "PER-I": 2, "PER-E": 3,
             "LOC-S": 4, "LOC-B": 5,
             "LOC-I": 6, "LOC-E": 7,
             "OFI-B": 8, "OFI-I": 9,
             "OFI-E":10, "PER-S":11,
             "OFI-S":11
             }
    else:
        tags = tags.split('/')

        for i in range(len(tags)):
            tag2id[tags[i]] = i
            
    for k,v in tag2id.items():
        id2tag[v] = k
    return tag2id,id2tag
    
    
    
#
# def fan_jian(char):
#     '''
#     繁简转化
#     '''
#
#     jian_char = Converter('zh-hans').convert(char)
#
#     return jian_char


def write_vector(path, content):
    with open(path, 'wb') as fw2:
        pickle.dump(content, fw2)


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    return word2id


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def one_hot(some2id):
    dim = len(some2id)
    _vec = []
    for i in range(dim):
        vec = [0] * dim
        vec[i] = 1
        _vec.append(vec)
    _vec[0] = [0] * dim
    return _vec


def load_bin_vec(fname, vocab, ksize=300):
    time_str = datetime.datetime.now().isoformat()
    print("{}:开始筛选w2v数据词汇...".format(time_str))
    word_vecs = {}
    model = {}
    # model = gensim.models.KeyedVectors.load_word2vec_format(fname,binary=False)
    # model = np.genfromtxt(fname)
    with open(fname, encoding='utf-8', errors='ignore') as f:
        firstline = True
        for line in f:
            if firstline:
                firstline = False
                dim = int(line.rstrip().split()[1])
                continue
            tokens = line.rstrip().split(' ')
            model[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    for word in vocab:
        try:
            word_vecs[word] = model[word]
        except:
            word_vecs[word] = [0.25] * ksize
    time_str = datetime.datetime.now().isoformat()
    print("{}:筛选w2v数据词汇结束...".format(time_str))
    return word_vecs


def get_W(word_vecs, vocab_ids_map, k=300, is_rand=False):
    time_str = datetime.datetime.now().isoformat()
    print("{}:生成嵌入层参数W...".format(time_str))
    vocab_size = len(word_vecs)
    W = np.random.uniform(-1.0, 1.0, size=[vocab_size, k]).astype(np.float32)
    print("非随机初始化...")
    # print("{}:词表ID".format(vocab_ids_map))
    for i, word in enumerate(word_vecs):
        id = vocab_ids_map[word]
        if id == 0:
            W[id] = [0] * k  # PAD
        elif id == len(vocab_ids_map) - 1:
            W[id] = [0.25] * k  # UNK
        else:
            W[id] = word_vecs[word]
    time_str = datetime.datetime.now().isoformat()
    print("{}:生成嵌入层参数W完毕".format(time_str))
    return W


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, pos_, dt_, tag_ = [], [], [], []
    for line in lines:
        if line != '\n':
            #[char, pos, dt, label] = line.strip().split('\t')
            [char, label] = line.strip().split('\t')
            sent_.append(char)
            tag_.append(label)
            # pos_.append(pos)
            # dt_.append(dt)
        else:
            # data.append((sent_, pos_, dt_, tag_))
            data.append((sent_, tag_))
            sent_, pos_, dt_, tag_ = [], [], [], []
    return data


def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id, ''.join(sent)


def pos2id(pos, pos2id):
    pos_id = []
    for word in pos:
        if word in pos2id.keys():
            pos_id.append(pos2id[word])
        else:
            pos_id.append(pos2id['<UNK>'])
    return pos_id


def add_pad_unk(some_list):
    '''
    在字表的开头加上PAD，末尾加上未登录词unk
    '''
    some_list = list(set(some_list))
    some_list.insert(0, '<PAD>')
    some_list.append('<UNK>')
    dim = len(some_list)
    some2id = dict(zip(some_list, range(dim)))
    return some2id


def voc_build(_path):
    """
    :param vocab_path:
    :return:
    """
    d_char = {}
    char_list, pos_list, dict_list = [], [], []
    #原始的训练文本有字／词性／字典信息
    # dirs = os.listdir('./data_path/guhanyu/')
    
        
    with open(_path) as fr:
        lines = fr.readlines()
    for line in lines:
        try:
                # char, pos_tag, dict_tag, label = line.strip().split('\t')
            char, label = line.strip().split('\t') 
                #承希师兄的文件汇总字符和tag之间是空格,我处理成'\t'
            if char not in d_char.keys():
                d_char[char] = char
                
                # pos_list.append(pos_tag)
    #             dict_list.append(dict_tag)
        except:
            if line != ' \n': #all_gu.txt中以空格加换行符来分隔句子
                print('this line wrong:',line)
    for key in d_char.keys():
        '''利用字典方便去重'''
        char_list.append(key)

    word2id = add_pad_unk(char_list)
    pos2id = add_pad_unk(pos_list)
    d2id = add_pad_unk(dict_list)
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_seq(sequences, dts, lenth):
    seq_list = []
    for i in range(len(sequences)):
        seq = list(sequences[i])
        seq_ = seq * len(dts[i]) + (lenth - len(dts[i])) * [0]
        seq_list.append(seq_)
    return seq_list


def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab,tag2label, shuffle=False):
    """
    将句子和tag都转成id，每次处理一个batch
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return: 
    """
    if shuffle:
        random.shuffle(data)
    seqs, labels,label_ = [], [], []
    
    for (sent_, tag_) in data:
        sent_, sen = sentence2id(sent_, vocab)

        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels, = [], []
        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs,labels
