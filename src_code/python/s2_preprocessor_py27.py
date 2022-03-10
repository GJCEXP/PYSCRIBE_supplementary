#coding=utf-8
import json
from config_py27 import *
from my_lib.util.code_parser.py_parser import py2ast,tokenize_python
from my_lib.util.code_parser.code_tokenizer import tokenize_code
from my_lib.util.code_parser.astor import MyAstor
import codecs
import os
from tqdm import tqdm
import re
import numpy as np

import ast

def cut_str_code(code,seg_word_dic,max_code_str_len=22):
    def _get_long_strs_in_code(code, min_len=22):
        nodes, _, poses = py2ast(code, attr='all', seg_attr=False)
        # long_strs=[]
        # for node,pos in zip(nodes,poses):
        #     if poses[-1]<0:
        long_strs = [node for node, pos in zip(nodes,poses) if pos[-1] < 0
                     and len(node.replace('\\n',' ').strip().split())>min_len]
        return long_strs

    long_strs=_get_long_strs_in_code(code,min_len=max_code_str_len)
    long_strs=sorted(long_strs,key=len,reverse=True)
    # print(code)
    for string in long_strs:
        # string=string[:-1]
        cut_str=' '.join(string.replace('\\n',' ').strip().split()[:max_code_str_len])
        # print(string)
        while True:
            str_tokens = tokenize_code(cut_str, user_words=USER_WORDS, lemmatize=True, lower=True, keep_punc=True,
                                    seg_var=True, err_dic=seg_word_dic)
            if len(str_tokens)>=max_code_str_len:
                cut_str=cut_str.split()
                cut_str=' '.join(cut_str[:len(cut_str)-1])
            break
        code=code.replace(string,cut_str)
    assert ast.parse(code)
    return code


def tokenize_raw_data(raw_data_path, token_data_path, seg_word_dic_path,
                      max_code_str_len=max_code_str_len,
                      max_ast_size=max_ast_size,
                      max_text_len=max_text_len,):
    logging.info('########### Start tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

    token_data_dir = os.path.dirname(token_data_path)
    if not os.path.exists(token_data_dir):
        os.makedirs(token_data_dir)
    with codecs.open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    with codecs.open(seg_word_dic_path,'r', encoding='utf-8') as f:
        seg_word_dic=json.load(f)
    # lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
    real_max_ast_size = 0
    real_max_code_len = 0
    real_max_text_len = 0
    for i,item in enumerate(tqdm(raw_data)):
        # logging.info('------Process the %d-th item' % (i + 1))
        # token_item['id']=item['id']
        text_tokens=tokenize_code(item['text'],user_words=USER_WORDS,lemmatize=True, lower=True,keep_punc=True,
                                  seg_var=True,err_dic=seg_word_dic)
        item['text']=' '.join(text_tokens[:max_text_len])
        item['text']=re.sub(r'\d+','<number>',item['text'])
        code = cut_str_code(item['code'], seg_word_dic=seg_word_dic, max_code_str_len=max_code_str_len)
        nodes,edge_index,node_poses=py2ast(code,attr='all',seg_attr=True,lemmatize=True,lower=True,keep_punc=True,
                                 seg_var=True,err_dic=seg_word_dic,user_words=USER_WORDS)
        nodes=nodes[:max_ast_size]
        node_poses=['({},{},{})'.format(node_pos[0], node_pos[1], node_pos[2]) for node_pos in node_poses[:max_ast_size]]
        edge_index = edge_index[:, :max_ast_size - 1]
        edges, edge_poses = [], []
        for i in range(edge_index.shape[1]):
            edges.append("({},{})".format(nodes[edge_index[1, i]], nodes[edge_index[0, i]]))
            edge_poses.append(node_poses[edge_index[0, i]])
        item['ast'] = {'nodes': str(nodes),
                       'edges': str(edges),
                       'node_poses': str(node_poses),
                       'edge_poses': str(edge_poses)}

        item['ast']['nodes']=re.sub(r'\d+','<number>',item['ast']['nodes'])
        # code_tokens = tokenize_python(code, lower=True,lemmatize=True, keep_punc=True,
        #                           seg_var=True,err_dic=seg_word_dic,user_words=USER_WORDS)
        # item['code'] = ' '.join(code_tokens[:max_code_len])
        # item['code'] = re.sub(r'\d+', '<number>', item['code'])

        real_max_ast_size = max(real_max_ast_size, len(list(eval(item['ast']['nodes']))))
        # real_max_code_len = max(real_max_code_len, len(item['code'].split()))
        real_max_text_len = max(real_max_text_len, len(item['text'].split()))
    logging.info('real_max_ast_size: {}, real_max_text_len: {}'.
                 format(real_max_ast_size,real_max_text_len))

    with codecs.open(token_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

if __name__=='__main__':
    tokenize_raw_data(raw_data_path=train_raw_data_path,
                      token_data_path=train_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      # max_code_len=max_code_len,
                      max_ast_size=max_ast_size)
    tokenize_raw_data(raw_data_path=valid_raw_data_path,
                      token_data_path=valid_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      # max_code_len=max_code_len,
                      max_ast_size=max_ast_size)
    tokenize_raw_data(raw_data_path=test_raw_data_path,
                      token_data_path=test_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      # max_code_len=max_code_len,
                      max_ast_size=max_ast_size)