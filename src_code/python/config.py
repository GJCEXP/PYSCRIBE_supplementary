#coding=utf-8
import logging
import os
import sys
from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
                                              get_nltk33_sent_bleu2 as get_sent_bleu2,  \
                                            get_nltk33_sent_bleu3 as get_sent_bleu3,  \
                                            get_nltk33_sent_bleu4 as get_sent_bleu4,  \
                                            get_nltk33_sent_bleu as get_sent_bleu
# from my_lib.util.eval.translate_metric import get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu
from my_lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
import math
from config_py27 import *

# fine_token_data_dir=os.path.join(top_data_dir,'fine_token_data/')
# train_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(train_data_name))
# valid_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(valid_data_name))
# test_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(test_data_name))

w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
ast_node_w2i_path=os.path.join(w2i2w_dir,'ast_node_w2i.pkl')
ast_node_i2w_path=os.path.join(w2i2w_dir,'ast_node_i2w.pkl')
ast_edge_w2i_path=os.path.join(w2i2w_dir,'ast_edge_w2i.pkl')
ast_edge_i2w_path=os.path.join(w2i2w_dir,'ast_edge_i2w.pkl')
ast_pos_w2i_path=os.path.join(w2i2w_dir,'ast_pos_w2i.pkl')
ast_pos_i2w_path=os.path.join(w2i2w_dir,'ast_pos_i2w.pkl')
text_w2i_path=os.path.join(w2i2w_dir,'text_w2i.pkl')
text_i2w_path=os.path.join(w2i2w_dir,'text_i2w.pkl')

in_min_token_count=3
out_min_token_count=3
unk_aliased=True  #是否将未知的rare tokens进行标号处理

avail_data_dir=os.path.join(top_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

OUT_BEGIN_TOKEN='</s>'
OUT_END_TOKEN='</e>'
PAD_TOKEN='<pad>'
UNK_TOKEN='<unk>'

model_dir=os.path.join(top_data_dir,'model/')
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" ####################"0,1,2,3,4,5","0,1,2,3"
import os
#

emb_dims = 512  ####################
ast_att_layers=4    ####################
text_att_layers=4   ####################
model_name='code2text_{}_{}_{}_{}'.format(ast_att_layers,ast_att_layers,text_att_layers,emb_dims)
params = dict(model_dir=model_dir,
              model_name=model_name,
              model_id=None,
              emb_dims=emb_dims,
              ast_att_layers=ast_att_layers,
              ast_att_heads=8,
              ast_att_head_dims=None,
              ast_ff_hid_dims=4 * emb_dims,
              text_att_layers=text_att_layers,
              text_att_heads=8,
              text_att_head_dims=None,
              text_ff_hid_dims=4 * emb_dims,
              drop_rate=0.2,
              pad_idx=0,
              train_batch_size=120,  ####################192
              pred_batch_size=math.ceil(120 * 4),  #####################192*2.5
              gpu0_train_batch_size=25,  ####################40
              max_train_size=-1,  ####################-1
              max_valid_size=math.ceil(120 * 4) * 10,  ####################10
              max_big_epochs=100, ####################100
              early_stop=10,
              regular_rate=1e-5,
              lr_base=5e-4,
              lr_decay=0.95,
              min_lr_rate=0.05,
              warm_big_epochs=3,
              beam_width=5,
              start_valid_epoch=50,  ####################40
              gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
              train_mode=True)

MAX_TEST_SIZE=-1   ####################-1

# from tmp_google_bleu import get_sent_bleu
train_metrics = [get_sent_bleu]
valid_metric = get_sent_bleu
test_metrics = [get_rouge, get_cider,get_meteor,
                get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu,
                get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu] #[get_corp_bleu]


#the path of result in practical prediction
res_dir=os.path.join(top_data_dir,'result/')
res_path=os.path.join(res_dir,model_name+'.json')

import random
import torch
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
seeds=[0,42,7,23,124,1084,87]
seed_torch(seeds[1])