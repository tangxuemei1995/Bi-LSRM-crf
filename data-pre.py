
'''
此代码用于处理语料
'''

import os
import random


def write_data(path,content):
    '''

    :param path:
    :param content: str
    :return:
    '''
    f = open(path, 'w', encoding='utf-8')
    f.write(content)
    f.close()
    print('write data finished!')


def write_data_1(path,content):
    '''

    :param path:
    :param content: list
    :return:
    '''
    f = open(path, 'w', encoding='utf-8')
    for li in content:
        f.write(li)
    f.close()
    print('write data finished!')

def data_seg(path):
    '''
    切分数据集

    :param path:
    :return:
    '''
    train, test, dev, all = [], [], [], []
    text = ''
    filename = os.listdir(path)
    for name in filename:
        print(name)
        for line in open(path + '/' + name):
            if line != ' \n':
                text += line

            else:
                text += '\n'
                all.append(text)
                text = ''

    print(len(all))
    id = [i for i in range(len(all))]
    random.shuffle(id)
    for index in id[0:len(id)//10]:
        test.append(all[index])
    for index in id[len(id)//10:(len(id)//10)*2]:
        dev.append(all[index])

    for index in id[(len(id)//10)*2::]:
        train.append(all[index])

    write_data_1('./data/dev.txt', dev)
    print('dev size:\t',len(dev))
    print('dev set finished!')
    write_data_1('./data/test.txt', test)
    print('test size:\t',len(test))
    print('test set finished!')
    write_data_1('./data/train.txt', train)
    print('train size:\t',len(train))
    print('train set finished!')


    return 0


def data_merge(path):
    '''
    :param path:
    :return:
    '''
    all_data= ''
    filename = os.listdir(path)
    for name in filename:
        print(name)
        for line in open(path + '/' + name):
            if line != '\n':
                line = line.split(' ')
                all_data += line[0] + '\t' + line[1]
            else:
                all_data += ' \n'
    write_data('./data/zi.txt',all_data)
    return 0


if __name__=="__main__":
    # print(1)
    # data_merge('./data/Zizhitongjian_ner')
    data_seg('./data/data')