import os

'''
此代码用于评测
'''
from metric import get_ner_fmeasure

def conlleval(label_predict, label_path, metric_path):
    """
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    gold_all = []
    pred_all = []

    for sent_result in label_predict:
        gold_one = []
        pred_one = []
        for char, tag, tag_ in sent_result:
            if tag != 'O':
                t, head = tag.split('-', 1)
                if head == 'M':
                    head = 'I'
                tag = head + '-' + t
            gold_one.append(tag)

            if tag_ == 0:
                tag_ = 'O'
            else:
                t_, head_ = tag_.split('-', 1)
                if head_ == 'M':
                    head_ = 'I'
                tag_ = head_ + '-' + t_
            pred_one.append(tag_)
        
        gold_all.append(gold_one)
        pred_all.append(pred_one)

    precision, recall, f_measure, acc = get_ner_fmeasure(gold_all, pred_all, "BIOES")

    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    # eval(label_path, metric_path)
    with open(metric_path,'w') as fr:
        fr.write('token_acc:' + '\t' + str(precision))
        fr.write('\npre:' + '\t' + str(precision))
        fr.write('\nrecall:' + '\t' + str(recall))
        fr.write('\nf1:' + '\t' + str(f_measure))
    return precision, recall, f_measure


def find_entity(label_path, i):
    d1, d = {}, {}
    entity = ''
    token_counter, correct_tags = 0, 0
    n = 1
    for line in open(label_path, "r", encoding='utf-8'):
        if line != '\n' and line != '':
            token_counter = token_counter+1
            list1 = line.strip().split(' ')
            if list1[len(list1)-i] == list1[len(list1)-2]:
                correct_tags = correct_tags + 1
            if list1[len(list1)-i].startswith('B'):
                entity = list1[0]
            elif list1[len(list1)-i].startswith('M'):
                if entity != '':
                    entity = entity + list1[0]
            elif list1[len(list1)-i].startswith('E'):
                if entity != '':
                    entity = entity + list1[0]
                    key = list1[len(list1)-i][2::].replace('\n', '')
                    d1[key] = entity
                    entity = ''
            elif list1[len(list1)-i].startswith('S'):
                entity = list1[0]
                key = list1[len(list1)-i][2::].replace('\n', '')
                d1[key] = entity
                entity = ''
        else:
            d[str(n)] = d1
            d1 = {}
            n += 1
    return d, token_counter, correct_tags


def eval(label_path, metric_path):
    correct_entity, all_entity, found_entity = 0, 0, 0
    d_pre, token_counter, correct_tags = find_entity(label_path, 1)
    d, _, _ = find_entity(label_path, 2)
    for i in range(len(d)):
        all_entity += len(d[str(i+1)])
        found_entity += len(d_pre[str(i+1)])
        if d[str(i+1)] == d_pre[str(i+1)]:
            correct_entity += len(d[str(i+1)])
        else:
            for k in d[str(i+1)].keys():
                if k in d_pre[str(i+1)].keys():
                    if d[str(i+1)][k] == d_pre[str(i+1)][k]:
                        correct_entity += 1
    print('correct tags:\t', correct_tags)
    print('all tags:\t', token_counter)
    print('correct entities:\t', correct_entity)
    print('all entities:\t', all_entity)
    print('found entities:\t', found_entity)
    acc = correct_tags/token_counter
    pre = correct_entity/found_entity
    recall = correct_entity/all_entity
    f1 = 2*pre*recall/(pre+recall)
    f = open(metric_path, 'w', encoding='utf-8')
    f.write('accuracy:' + '\t' + str(acc) + '\n')
    f.write('\t'+'precision:' + '\t' + str(pre) + '\n')
    f.write('\t'+'recall:' + '\t' + str(recall) + '\n')
    f.write('\t'+'f1:' + '\t' + str(f1) + '\n')
