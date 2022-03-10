#coding=utf-8
import pickle
from collections import Counter
import nltk
import enchant
import json
from config import *
from my_lib.util.code_parser.java_parser import java2ast,tokenize_java
from my_lib.util.code_parser.code_tokenizer import tokenize_code
from my_lib.util.code_parser.astor import MyAstor
import codecs
import os
from tqdm import tqdm
import re
import javalang

def _seg_conti_word(word_str, user_words=None):
    """
    Segment a string of word_str using the pyenchant vocabulary.
    Keeps longest possible words that account for all characters,
    and returns list of segmented words.

    :param word_str: (str) The character string to segment.
    :param exclude: (set) A set of string to exclude from consideration.
                    (These have been found previously to lead to dead ends.)
                    If an excluded word occurs later in the string, this
                    function will fail.
    """
    def _seg_with_digit(words):
        text=' '.join(words)
        digits = re.findall(r'\d+', text)
        digits = sorted(list(set(digits)), key=len, reverse=True)
        # digit_str = ''
        for digit in digits:
            text = text.replace(digit, ' ' + digit + ' ')
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        tokens=[lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a') for word
                in text.strip().split()]
        if len(tokens)>0:
            new_tokens=[tokens[0]]
            for token in tokens[1:]:
                if new_tokens[-1].isdigit() and token.isdigit():
                    new_tokens[-1]+=token
                else:
                    new_tokens.append(token)
            tokens=new_tokens
        return tokens

    if not user_words:
        user_words = set()
    try:
        eng_dict = enchant.Dict("en_US")
        if eng_dict.check(word_str):
            return [word_str]

        # if not word_str[0].isalpha():  # don't check punctuation etc.; needs more work
        #     return [word_str]

        left_words,right_words = [],[]

        working_chars = word_str
        while working_chars:
            # iterate through segments of the word_str starting with the longest segment possible
            for i in range(len(working_chars), 2, -1):
                left_chars = working_chars[:i]
                if left_chars in user_words or eng_dict.check(left_chars):
                    left_words.append(left_chars)
                    working_chars = working_chars[i:]
                    user_words.add(left_chars)
                    break
            else:
                for i in range(0, len(working_chars) - 2):
                    right_chars = working_chars[i:]
                    if right_chars in user_words or eng_dict.check(right_chars):
                        right_words.insert(0, right_chars)
                        working_chars = working_chars[:i]
                        user_words.add(right_chars)
                        break
                else:
                    return _seg_with_digit(left_words+[working_chars]+right_words)

        if working_chars!='':
            return _seg_with_digit(left_words + [working_chars] + right_words)
        else:
            return _seg_with_digit(left_words + right_words)
    except Exception:
        return _seg_with_digit([word_str])

def make_seg_word_dict(train_raw_data_path,valid_raw_data_path,test_raw_data_path,seg_word_dic_path):
    logging.info('Start making segmented word dictionary.')

    path_dir = os.path.dirname(seg_word_dic_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(train_raw_data_path,'r') as f1,open(valid_raw_data_path,'r') as f2,open(test_raw_data_path,'r') as f3:
        train_raw_data,valid_raw_data,test_raw_data=json.load(f1),json.load(f2),json.load(f3)
    # token_counter=Counter()
    token_set=set()
    for raw_data in [train_raw_data,valid_raw_data,test_raw_data]:
        pbar = tqdm(raw_data)
        pbar.set_description('[Extract tokens]')
        for item in pbar:
            code_tokens = tokenize_java(item['code'], lower=True, lemmatize=True, keep_punc=True,
                                          seg_var=True,err_dic=None,user_words=USER_WORDS)
            # print(USER_WORDS)
            # pbar.set_description('[Extract tokens, length: {}]'.format(len(code_tokens)))
            text_tokens = tokenize_code(item['text'], user_words=USER_WORDS, lemmatize=True, lower=True,keep_punc=True,
                                        seg_var=True,err_dic=None)
            token_set |= set(code_tokens)
            token_set |= set(text_tokens)
    seg_token_dic=dict()
    pbar=tqdm(token_set)
    pbar.set_description('[Segment tokens]')
    user_words = set(['mk', 'dir', 'json', 'config', 'html', 'arange', 'bool', 'eval','mod','boolean'])
    # seg_count=0
    for token in pbar:
        seg_token=' '.join(_seg_conti_word(token,user_words=user_words))
        if seg_token != token:
            # seg_count+=1
            seg_token_dic[token] = seg_token
            pbar.set_description('[Segment tokens: {}-th segmented: {}:::{}]'.format(len(seg_token_dic),token,seg_token))

    with codecs.open(seg_word_dic_path,'w',encoding='utf-8') as f:
        json.dump(seg_token_dic,f,indent=4, ensure_ascii=False)
    logging.info('Finish making segmented word dictionary.')

def cut_str_code(code,seg_word_dic,max_code_str_len=22):
    def _get_long_strs_in_code(code, min_len=22):
        nodes, _, poses = java2ast(code, attr='all', seg_attr=False)
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
    assert javalang.parse.parse(code)
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
        code = '{}\n{}\n{}'.format('class _ {', item['code'], '}')
        code = cut_str_code(code, seg_word_dic=seg_word_dic, max_code_str_len=max_code_str_len)
        nodes,edge_index,node_poses=java2ast(code,attr='all',seg_attr=True,lemmatize=True,lower=True,keep_punc=True,
                                 seg_var=True,err_dic=seg_word_dic,user_words=USER_WORDS)
        nodes=nodes[:max_ast_size]
        node_poses=['({},{},{})'.format(node_pos[0], node_pos[1], node_pos[2]) for node_pos in node_poses[:max_ast_size]]
        edge_index = edge_index[:, :max_ast_size - 1]
        edges, edge_poses = [], []
        for i in range(edge_index.shape[1]):
            edges.append('({},{})'.format(nodes[edge_index[1, i]], nodes[edge_index[0, i]]))
            edge_poses.append(node_poses[edge_index[0, i]])
        item['ast'] = {'nodes': str(nodes),
                       'edges': str(edges),
                       'node_poses': str(node_poses),
                       'edge_poses': str(edge_poses)}

        item['ast']['nodes']=re.sub(r'\d+','<number>',item['ast']['nodes'])
        # code_tokens = tokenize_java(code, lower=True,lemmatize=True, keep_punc=True,
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
    make_seg_word_dict(train_raw_data_path, valid_raw_data_path, test_raw_data_path, seg_word_dic_path)

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