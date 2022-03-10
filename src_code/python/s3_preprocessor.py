#coding=utf-8
import json
import pickle
from config import *
from collections import Counter
import sys
import codecs
from tqdm import tqdm
from my_lib.util.code_parser.astor import MyAstor
import enchant  # pip install pyenchant
import re
import nltk

def build_w2i2w(train_token_data_path,
                ast_node_w2i_path,
                ast_node_i2w_path,
                ast_edge_w2i_path,
                ast_edge_i2w_path,
                ast_pos_w2i_path,
                ast_pos_i2w_path,
                text_w2i_path,
                text_i2w_path,
                in_min_token_count=3,
                out_min_token_count=3,
                unk_aliased=True,
                ):
    logging.info('########### Start building the dictionary of the training set ##########')
    dic_paths = [ast_node_w2i_path,
                 ast_node_i2w_path,
                 ast_edge_w2i_path,
                 ast_edge_i2w_path,
                 ast_pos_w2i_path,
                 ast_pos_i2w_path,
                 text_w2i_path,
                 text_i2w_path,
                 ]
    for dic_path in dic_paths:
        dic_dir = os.path.dirname(dic_path)
        if not os.path.exists(dic_dir):
            os.makedirs(dic_dir)

    with codecs.open(train_token_data_path, 'r', encoding='utf-8') as f:
        token_data = json.load(f)

    ast_node_counter = Counter()
    ast_edge_counter = Counter()
    ast_pos_counter = Counter()
    text_token_counter = Counter()
    max_ast_size=0
    max_text_len=0
    for i, item in enumerate(tqdm(token_data)):
        ast_nodes=list(eval(item['ast']['nodes']))
        ast_node_counter += Counter(ast_nodes)
        ast_edges = list(eval(item['ast']['edges']))
        ast_edge_counter += Counter(ast_edges)
        ast_poses=list(eval(item['ast']['node_poses']))+list(eval(item['ast']['edge_poses']))
        ast_pos_counter += Counter(ast_poses)
        text_token_counter += Counter(item['text'].split())  # texts是一个列表
        max_ast_size = max(max_ast_size, len(list(eval(item['ast']['nodes']))))
        max_text_len=max(max_text_len,len(item['text'].split()))
    logging.info('max_ast_size: {},max_text_len: {}'.format(max_ast_size,max_text_len))
    general_vocabs = [PAD_TOKEN, UNK_TOKEN]

    ast_nodes = list(filter(lambda x: ast_node_counter[x] >= in_min_token_count, ast_node_counter.keys()))
    ast_edges = list(filter(lambda x: ast_edge_counter[x] >= in_min_token_count, ast_edge_counter.keys()))
    unk_node_aliases=[]
    unk_edge_aliases=[]
    if unk_aliased:
        max_node_alias_num = 0
        max_edge_alias_num = 0
        for i, item in enumerate(token_data):
            node_aliases = list(filter(lambda x: x not in ast_nodes, set(list(eval(item['ast']['nodes'])))))
            edge_aliases = list(filter(lambda x: x not in ast_edges, set(list(eval(item['ast']['edges'])))))
            max_node_alias_num = max(max_node_alias_num, len(node_aliases))
            max_edge_alias_num = max(max_edge_alias_num, len(edge_aliases))
        unk_node_aliases = ['<unk-alias-{}>'.format(i) for i in range(max_node_alias_num)]
        unk_edge_aliases = ['<unk-alias-{}>'.format(i) for i in range(max_edge_alias_num)]
    ast_nodes = general_vocabs + ast_nodes+unk_node_aliases
    ast_edges = general_vocabs + ast_edges+unk_edge_aliases

    ast_poses = list(filter(lambda x: ast_pos_counter[x] >= in_min_token_count, ast_pos_counter.keys()))
    ast_poses = general_vocabs + ast_poses

    text_tokens = list(filter(lambda x: text_token_counter[x] >= out_min_token_count, text_token_counter.keys()))
    text_tokens = general_vocabs + text_tokens + [OUT_END_TOKEN, OUT_BEGIN_TOKEN,]

    ast_node_indices = list(range(len(ast_nodes)))
    ast_edge_indices = list(range(len(ast_edges)))
    ast_pos_indices = list(range(len(ast_poses)))
    text_indices = list(range(len(text_tokens)))

    ast_node_w2i = dict(zip(ast_nodes, ast_node_indices))
    ast_node_i2w = dict(zip(ast_node_indices, ast_nodes))
    ast_edge_w2i = dict(zip(ast_edges, ast_edge_indices))
    ast_edge_i2w = dict(zip(ast_edge_indices, ast_edges))
    ast_pos_w2i = dict(zip(ast_poses, ast_pos_indices))
    ast_pos_i2w = dict(zip(ast_pos_indices, ast_poses))
    text_w2i = dict(zip(text_tokens, text_indices))
    text_i2w = dict(zip(text_indices, text_tokens))

    dics = [ast_node_w2i,
            ast_node_i2w,
            ast_edge_w2i,
            ast_edge_i2w,
            ast_pos_w2i,
            ast_pos_i2w,
            text_w2i,
            text_i2w]
    for dic, dic_path in zip(dics, dic_paths):
        with open(dic_path, 'wb') as f:
            pickle.dump(dic, f)
        with codecs.open(dic_path + '.json', 'w') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish building the dictionary of the training set ##########')

def build_avail_data(token_data_path,
                     avail_data_path,
                     ast_node_w2i_path,
                     ast_edge_w2i_path,
                     ast_pos_w2i_path,
                     text_w2i_path,
                     unk_aliased=True):
    '''
    根据字典构建模型可用的数据集，数据集为一个列表，每个元素为一条数据，是由输入和输出两个元素组成的，
    输入元素为一个ndarray，每行分别为边起点、边终点、深度、全局位置、局部位置，
    输出元素为一个ndarray，为输出的后缀表达式
    :param token_data_path:
    :param avail_data_path:
    :param ast_node_w2i_path:
    :param edge_depth_w2i_path:
    :param edge_lpos_w2i_path:
    :param edge_spos_w2i_path:
    :return:
    '''
    logging.info('########### Start building the train dataset available for the model ##########')
    avail_data_dir = os.path.dirname(avail_data_path)
    if not os.path.exists(avail_data_dir):
        os.makedirs(avail_data_dir)

    w2is=[]
    for w2i_path in [ast_node_w2i_path,
                     ast_edge_w2i_path,
                     ast_pos_w2i_path,
                     text_w2i_path,
                     text_i2w_path]:
        with open(w2i_path,'rb') as f:
            w2is.append(pickle.load(f))
    ast_node_w2i,ast_edge_w2i,ast_pos_w2i,text_w2i,text_i2w=w2is

    logging.info('We have {} node tokens, {} edge tokens, {} ast_pos tokens, {} text tokens'.
                 format(len(ast_node_w2i),len(ast_edge_w2i),len(ast_pos_w2i),len(text_w2i)))
    unk_idx = w2is[0][UNK_TOKEN]
    pad_idx=w2is[0][PAD_TOKEN]
    with codecs.open(token_data_path,'r') as f:
        token_data=json.load(f)

    avail_data={'asts':[],'texts':[]}
    text_token_idx_counter=Counter()
    max_ast_size = 0
    max_text_len = 0
    pbar=tqdm(token_data)
    for i,item in enumerate(pbar):
        ast_nodes = list(eval(item['ast']['nodes']))
        ast_edges = list(eval(item['ast']['edges']))
        ast_node_poses=list(eval(item['ast']['node_poses']))
        ast_edge_poses=list(eval(item['ast']['edge_poses']))
        text_tokens=item['text'].split()

        if unk_aliased:
            all_unk_node_aliases = filter(lambda x: x not in ast_node_w2i.keys(), ast_nodes)
            unk_node_aliases=[]
            for unk_alias in all_unk_node_aliases:
                if unk_alias not in unk_node_aliases:
                    unk_node_aliases.append(unk_alias)
            all_unk_edge_aliases = filter(lambda x: x not in ast_edge_w2i.keys(), ast_edges)
            unk_edge_aliases = []
            for unk_alias in all_unk_edge_aliases:
                if unk_alias not in unk_edge_aliases:
                    unk_edge_aliases.append(unk_alias)
            ast_nodes = [node if node not in unk_node_aliases else '<unk-alias-{}>'.format(unk_node_aliases.index(node)) for node in ast_nodes]
            ast_edges = [edge if edge not in unk_edge_aliases else '<unk-alias-{}>'.format(unk_edge_aliases.index(edge)) for edge in ast_edges]

        ast_node_ids=[ast_node_w2i.get(node,unk_idx) for node in ast_nodes]
        ast_edge_ids=[ast_edge_w2i.get(edge,unk_idx) for edge in ast_edges]
        ast_node_pos_ids=[ast_pos_w2i.get(pos,unk_idx) for pos in ast_node_poses]
        ast_edge_pos_ids=[ast_pos_w2i.get(pos,unk_idx) for pos in ast_edge_poses]
        text_token_ids=[text_w2i.get(token,unk_idx) for token in text_tokens]
        assert len(ast_node_ids)-1==len(ast_edge_ids)
        avail_item_in = {'nodes': ast_node_ids,
                         'edges': ast_edge_ids,
                         'node_poses': ast_node_pos_ids,
                         'edge_poses': ast_edge_pos_ids,
                         }
        avail_data['asts'].append(avail_item_in)
        avail_data['texts'].append(text_token_ids)

        max_ast_size = max(max_ast_size, len(ast_nodes))
        max_text_len = max(max_text_len, len(text_token_ids))
        text_token_idx_counter += Counter(text_token_ids)
    logging.info('max_ast_size: {}, max_text_len: {}'.format(max_ast_size,max_text_len))

    logging.info('+++++++++ The ratio of unknown text tokens is:%f' %(text_token_idx_counter[unk_idx]/sum(text_token_idx_counter.values())))
    with open(avail_data_path,'wb') as f:
        pickle.dump(avail_data,f)
    logging.info('########### Finish building the train dataset available for the model ##########')

if __name__=='__main__':
    build_w2i2w(train_token_data_path=train_token_data_path,
                ast_node_w2i_path=ast_node_w2i_path,
                ast_node_i2w_path=ast_node_i2w_path,
                ast_edge_w2i_path=ast_edge_w2i_path,
                ast_edge_i2w_path=ast_edge_i2w_path,
                ast_pos_w2i_path=ast_pos_w2i_path,
                ast_pos_i2w_path=ast_pos_i2w_path,
                text_w2i_path=text_w2i_path,
                text_i2w_path=text_i2w_path,
                in_min_token_count=in_min_token_count,
                out_min_token_count=out_min_token_count,
                unk_aliased=unk_aliased)

    build_avail_data(token_data_path=train_token_data_path,
                     avail_data_path=train_avail_data_path,
                     ast_node_w2i_path=ast_node_w2i_path,
                     ast_edge_w2i_path=ast_edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     unk_aliased=unk_aliased)

    build_avail_data(token_data_path=valid_token_data_path,
                     avail_data_path=valid_avail_data_path,
                     ast_node_w2i_path=ast_node_w2i_path,
                     ast_edge_w2i_path=ast_edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     unk_aliased=unk_aliased)

    build_avail_data(token_data_path=test_token_data_path,
                     avail_data_path=test_avail_data_path,
                     ast_node_w2i_path=ast_node_w2i_path,
                     ast_edge_w2i_path=ast_edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     unk_aliased=unk_aliased)
