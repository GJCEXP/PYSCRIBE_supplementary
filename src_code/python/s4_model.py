#coding=utf-8
import os
import pandas as pd
from my_lib.neural_module.learn_strategy import LrWarmUp
from my_lib.neural_module.transformer import TranEnc,DualTranDec
from my_lib.neural_module.embedding import PosEnc
from my_lib.neural_module.loss import LabelSmoothSoftmaxCEV2,CriterionNet
from my_lib.neural_module.balanced_data_parallel import BalancedDataParallel
from my_lib.neural_module.beam_search import trans_beam_search
from my_lib.neural_model.seq_to_seq_model import TransSeq2Seq
from my_lib.neural_model.base_model import BaseNet

from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset,DataLoader
import random
import numpy as np
import os
import logging
import pickle
import math
import codecs
from tqdm import tqdm
import json
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Datasetx(Dataset):
    '''
    文本对数据集对象（根据具体数据再修改）
    '''
    def __init__(self,asts,texts=None,ast_max_size=None,text_max_len=None,text_begin_idx=1,text_end_idx=2,pad_idx=0):
        self.len=len(asts)  #样本个数
        self.ast_max_size=ast_max_size
        self.text_max_len=text_max_len
        self.text_begin_idx=text_begin_idx
        self.text_end_idx=text_end_idx
        if ast_max_size is None:
            self.ast_max_size = max([len(item['nodes']) for item in asts])   #每个输入有多类特征
        if text_max_len is None and texts is not None:
            self.text_max_len=max([len(text) for text in texts]) #每个输出只是一个序列
        self.asts=asts
        self.texts=texts
        self.pad_idx=pad_idx
    def __getitem__(self, index):
        ast_nodes=self.asts[index]['nodes'][:self.ast_max_size]  # 先做截断
        ast_nodes = np.lib.pad(ast_nodes,
                               (0, self.ast_max_size - len(ast_nodes)),
                               'constant',
                               constant_values=(self.pad_idx, self.pad_idx))  # padding
        ast_edges = self.asts[index]['edges'][:self.ast_max_size-1]  # 先做截断
        ast_edges = np.lib.pad(ast_edges,
                               (0, self.ast_max_size -1 - len(ast_edges)),
                               'constant',
                               constant_values=(self.pad_idx, self.pad_idx))  # padding
        ast_node_poses = self.asts[index]['node_poses'][:self.ast_max_size]  # 先做截断
        ast_node_poses = np.lib.pad(ast_node_poses,
                               (0, self.ast_max_size - len(ast_node_poses)),
                               'constant',
                               constant_values=(self.pad_idx, self.pad_idx))  # padding
        ast_edge_poses = self.asts[index]['edge_poses'][:self.ast_max_size - 1]  # 先做截断
        ast_edge_poses = np.lib.pad(ast_edge_poses,
                               (0, self.ast_max_size - 1 - len(ast_edge_poses)),
                               'constant',
                               constant_values=(self.pad_idx, self.pad_idx))  # padding

        # tru_out_inputs=[]
        if self.texts is None:
            text_input = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
            text_input[0] = self.text_begin_idx
            return torch.tensor(ast_nodes), \
                   torch.tensor(ast_node_poses), \
                   torch.tensor(ast_edges), \
                   torch.tensor(ast_edge_poses), \
                   torch.tensor(text_input).long()
        else:
            text_output = self.texts[index][:self.text_max_len]  # 先做截断
            text_input=np.lib.pad(text_output,(1,self.text_max_len-len(text_output)),'constant',constant_values=(self.text_begin_idx, self.pad_idx))
            text_output=np.lib.pad(text_output, (0,1),'constant', constant_values=(self.pad_idx, self.text_end_idx))  # padding
            text_output= np.lib.pad(text_output, (0, self.text_max_len+1 - len(text_output)),
                                      'constant', constant_values=(self.pad_idx, self.pad_idx))  # padding
            # text_input=np.lib.pad(text_output[:-1],(1,0),'constant',constant_values=(self.text_begin_idx, 0))
            return torch.tensor(ast_nodes), \
                   torch.tensor(ast_node_poses), \
                   torch.tensor(ast_edges), \
                   torch.tensor(ast_edge_poses), \
                   torch.tensor(text_input).long(), \
                   torch.tensor(text_output).long()

    def __len__(self):
        return self.len

class Enc(nn.Module):
    def __init__(self,
                 ast_node_voc_size,
                 ast_edge_voc_size,
                 ast_pos_voc_size,
                 emb_dims=300,
                 att_layers=6,
                 att_heads=10,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        self.emb_dims = emb_dims
        self.node_embedding = nn.Embedding(ast_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        self.node_pos_embedding = nn.Embedding(ast_pos_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        self.edge_embedding = nn.Embedding(ast_edge_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        self.edge_pos_embedding=nn.Embedding(ast_pos_voc_size,emb_dims,padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(self.node_embedding.weight[1:, ])
        nn.init.xavier_uniform_(self.node_pos_embedding.weight[1:, ])
        nn.init.xavier_uniform_(self.edge_embedding.weight[1:, ])
        nn.init.xavier_uniform_(self.edge_pos_embedding.weight[1:, ])

        self.node_emb_layer_norm = nn.LayerNorm(emb_dims, elementwise_affine=True)
        self.edge_emb_layer_norm = nn.LayerNorm(emb_dims, elementwise_affine=True)
        self.node_emb_dropout = nn.Dropout(p=drop_rate)
        self.edge_emb_dropout = nn.Dropout(p=drop_rate)

        self.node_enc=TranEnc(query_dims=emb_dims,
                              head_num=att_heads,
                              head_dims=att_head_dims,
                              layer_num=att_layers,
                              ff_hid_dims=ff_hid_dims,
                              drop_rate=drop_rate)
        self.edge_enc = TranEnc(query_dims=emb_dims,
                                head_num=att_heads,
                                head_dims=att_head_dims,
                                layer_num=att_layers,
                                ff_hid_dims=ff_hid_dims,
                                drop_rate=drop_rate)


    def forward(self, ast_node,ast_node_pos,ast_edge,ast_edge_pos):
        '''

        :param x: [B,5,L1]
        :return:
        '''
        #encoding:
        node_emb=self.node_embedding(ast_node)*np.sqrt(self.emb_dims)   #(B,L_x,D)
        node_pos_emb=self.node_pos_embedding(ast_node_pos)  #(B,L_x,D)
        node_encoder=self.node_emb_dropout(node_emb.add(node_pos_emb))
        node_encoder=self.node_emb_layer_norm(node_encoder)
        node_mask = ast_node.abs().sign()  # (B,L)
        node_encoder=self.node_enc(query=node_encoder,query_mask=node_mask)    #(B,L_x,D)

        # print(torch.max(ast_edge),torch.max(ast_edge_pos))
        edge_emb = self.edge_embedding(ast_edge)*np.sqrt(self.emb_dims)  # (B,L_x,D)
        edge_pos_emb = self.edge_pos_embedding(ast_edge_pos)  # (B,L_x,D)
        # print(edge_emb.size(),edge_pos_emb.size())
        edge_encoder = self.edge_emb_dropout(edge_emb.add(edge_pos_emb))
        edge_encoder = self.edge_emb_layer_norm(edge_encoder)
        edge_mask = ast_edge.abs().sign()  # (B,L)
        edge_encoder = self.edge_enc(query=edge_encoder, query_mask=edge_mask)  # (B,L_x,D)

        return node_encoder,edge_encoder  #(B,out_dim,L_y)

class Dec(nn.Module):
    def __init__(self,
                 emb_dims,
                 text_voc_size,
                 text_max_len,
                 enc_out_dims,
                 att_layers,
                 att_heads,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.emb_dims = emb_dims
        self.text_voc_size = text_voc_size
        self.text_embedding = nn.Embedding(text_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(self.text_embedding.weight[1:, ])
        self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])  #不要忘了+1,因为输入前加了begin_id
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        # self.text_dec = TranDec(query_dims=emb_dims,
        #                         key_dims=enc_out_dims,
        #                         head_num=att_heads,
        #                         ff_hid_dims=ff_hid_dims,
        #                         head_dims=att_head_dims,
        #                         layer_num=att_layers,
        #                         drop_rate=drop_rate,
        #                         pad_idx=kwargs['pad_idx'],
        #                         self_causality=True)
        self.text_dec = DualTranDec(query_dims=emb_dims,
                                   key_dims=enc_out_dims,
                                   head_nums=att_heads,
                                   head_dims=att_head_dims,
                                   layer_num=att_layers,
                                   ff_hid_dims=ff_hid_dims,
                                   drop_rate=drop_rate,
                                   pad_idx=kwargs['pad_idx'],
                                   mode='sequential')
        self.dropout = nn.Dropout(p=drop_rate)
        self.out_fc = nn.Linear(emb_dims, text_voc_size)

    def forward(self,ast_node_enc,ast_edge_enc,text_input):
        text_emb = self.text_embedding(text_input) * np.sqrt(self.emb_dims)  # (B,L_text,D_text_emb)
        pos_emb = self.pos_encoding(text_input)  # # (B,L_text,D_emb)
        text_dec = self.dropout(text_emb.add(pos_emb))  # (B,L_text,D_emb)
        text_dec = self.emb_layer_norm(text_dec)  # (B,L_text,D_emb)

        ast_node_mask = ast_node_enc.abs().sum(-1).sign()  # (B,L_diff)
        ast_edge_mask = ast_edge_enc.abs().sum(-1).sign()  # (B,L_diff)
        text_mask = text_input.abs().sign()  # (B,L_text)
        text_dec = self.text_dec(query=text_dec,
                                 key1=ast_node_enc,
                                 key2=ast_edge_enc,
                                 query_mask=text_mask,
                                 key_mask1=ast_node_mask,
                                 key_mask2=ast_edge_mask,
                                 )  # (B,L_text,D_text_emb)

        text_output = self.out_fc(text_dec)  # (B,L_text,text_voc_size)包含begin_idx和end_idx
        return text_output.transpose(1, 2)

class TransNet(BaseNet):
    def __init__(self,
                 ast_max_size,
                 text_max_len,
                 ast_node_voc_size,
                 ast_edge_voc_size,
                 ast_pos_voc_size,
                 text_voc_size,
                 emb_dims=512,
                 ast_att_layers=6,
                 ast_att_heads=10,
                 ast_att_head_dims=None,
                 ast_ff_hid_dims=2048,
                 text_att_layers=6,
                 text_att_heads=10,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                    ** kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.init_params = locals()
        self.enc=Enc(ast_node_voc_size=ast_node_voc_size,
                         ast_edge_voc_size=ast_edge_voc_size,
                         ast_pos_voc_size=ast_pos_voc_size,
                         emb_dims=emb_dims,
                         att_layers=ast_att_layers,
                         att_heads=ast_att_heads,
                         att_head_dims=ast_att_head_dims,
                         ff_hid_dims=ast_ff_hid_dims,
                         drop_rate=drop_rate,
                         pad_idx=kwargs['pad_idx'])
        self.dec=Dec(emb_dims=emb_dims,
                         text_voc_size=text_voc_size,
                         text_max_len=text_max_len,
                         enc_out_dims=emb_dims,
                         att_layers=text_att_layers,
                         att_heads=text_att_heads,
                         att_head_dims=text_att_head_dims,
                         ff_hid_dims=text_ff_hid_dims,
                         drop_rate=drop_rate,
                         pad_idx=kwargs['pad_idx'])


    def forward(self, ast_node,ast_node_pos,ast_edge,ast_edge_pos,text_input):
        '''

        :param x: [B,5,L1]
        :param y: [B,L2]
        :return:
        '''
        #encoding:
        ast_node_enc,ast_edge_enc=self.enc(ast_node,ast_node_pos,ast_edge,ast_edge_pos)
        #decoding:
        text_output=self.dec(ast_node_enc,ast_edge_enc,text_input)
        return text_output  #(B,out_dim,L_y)

class TModel(TransSeq2Seq):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 emb_dims=512,
                 ast_att_layers=3,
                 ast_att_heads=8,
                 ast_att_head_dims=None,
                 ast_ff_hid_dims=2048,
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 pad_idx=0,
                 train_batch_size=32,
                 pred_batch_size=32,
                 gpu0_train_batch_size=0,
                 max_train_size=-1,
                 max_valid_size=32 * 10,
                 max_big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 start_valid_epoch=20,
                 early_stop=20,
                 Net=TransNet,
                 Dataset=Datasetx,
                 beam_width=1,
                 train_metrics=[get_sent_bleu],
                 valid_metric=get_sent_bleu,
                 test_metrics=[get_sent_bleu],
                 train_mode=True,
                 **kwargs
                 ):
        logging.info('Construct %s' % model_name)
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)
        self.init_params = locals()
        self.emb_dims = emb_dims
        self.ast_att_layers = ast_att_layers
        self.ast_att_heads = ast_att_heads
        self.ast_att_head_dims = ast_att_head_dims
        self.ast_ff_hid_dims = ast_ff_hid_dims
        self.text_att_layers = text_att_layers
        self.text_att_heads = text_att_heads
        self.text_att_head_dims = text_att_head_dims
        self.text_ff_hid_dims = text_ff_hid_dims
        self.drop_rate = drop_rate
        self.pad_idx = pad_idx
        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.gpu0_train_batch_size = gpu0_train_batch_size
        self.max_train_size = max_train_size
        self.max_valid_size = max_valid_size
        self.max_big_epochs = max_big_epochs
        self.regular_rate = regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate = min_lr_rate
        self.warm_big_epochs = warm_big_epochs
        self.start_valid_epoch=start_valid_epoch
        self.early_stop=early_stop
        self.Net = Net
        self.Dataset = Dataset
        self.beam_width = beam_width
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        self.test_metrics = test_metrics
        self.train_mode = train_mode

    def _logging_paramerter_num(self):
        logging.info("{} have {} paramerters in total".format(self.model_name, sum(
            x.numel() for x in self.net.parameters() if x.requires_grad)))
        # 计算enc+dec的parameter总数
        ast_enc_param_num = sum(x.numel() for x in self.net.module.enc.node_enc.parameters() if x.requires_grad) + \
                            sum(x.numel() for x in self.net.module.enc.edge_enc.parameters() if x.requires_grad)
        text_dec_param_num = sum(x.numel() for x in self.net.module.dec.text_dec.parameters() if x.requires_grad)
        enc_dec_param_num = ast_enc_param_num + text_dec_param_num
        logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

    def fit(self,
            train_data,
            valid_data,
            text_i2w,
            **kwargs
            ):
        self.ast_max_size = 0
        self.ast_node_voc_size = 0
        self.ast_edge_voc_size = 0
        self.ast_pos_voc_size = 0
        self.text_max_len=0
        self.text_voc_size=0
        for ast_item,text in zip(train_data['asts'],train_data['texts']):
            self.ast_max_size = max(self.ast_max_size,len(ast_item['nodes']))
            self.ast_node_voc_size = max(self.ast_node_voc_size,max(ast_item['nodes']))
            self.ast_edge_voc_size = max(self.ast_edge_voc_size,max(ast_item['edges']))
            self.ast_pos_voc_size = max(self.ast_pos_voc_size,max(ast_item['node_poses']+ast_item['edge_poses']))
            self.text_max_len=max(self.text_max_len,len(text))
            self.text_voc_size=max(self.text_voc_size,max(text))
        self.ast_node_voc_size+=1
        self.ast_edge_voc_size+=1
        self.ast_pos_voc_size+=1
        self.text_voc_size+=3
        assert self.text_voc_size==len(text_i2w)

        net = self.Net(emb_dims=self.emb_dims,
                       ast_max_size=self.ast_max_size,
                       text_max_len=self.text_max_len,
                       ast_node_voc_size=self.ast_node_voc_size,
                       ast_edge_voc_size=self.ast_edge_voc_size,
                       ast_pos_voc_size=self.ast_pos_voc_size,
                       text_voc_size=self.text_voc_size,
                       ast_att_layers=self.ast_att_layers,
                       ast_att_heads=self.ast_att_heads,
                       ast_att_head_dims=self.ast_att_head_dims,
                       ast_ff_hid_dims=self.ast_ff_hid_dims,
                       text_att_layers=self.text_att_layers,
                       text_att_heads=self.text_att_heads,
                       text_att_head_dims=self.text_att_head_dims,
                       text_ff_hid_dims=self.text_ff_hid_dims,
                       drop_rate=self.drop_rate,
                       pad_idx=self.pad_idx,
                       )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) == 1:
            self.net = nn.DataParallel(net.to(device))  # 并行使用GPU
        elif len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) > 1:
            self.net = BalancedDataParallel(self.gpu0_train_batch_size, net.to(device), dim=0)  # 并行使用多GPU

        self._logging_paramerter_num()  #需要有并行的self.net和self.model_name
        # self.net =DataParallel(net.to(device),follow_batch=['x'])  # 并行使用多GPU
        # self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
        # self.net = net.to(device)  # 数据转移到设备

        self.net.train()  # 设置网络为训练模式

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr_base,
                                    weight_decay=self.regular_rate)

        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
        # self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

        self.text_begin_idx = self.text_voc_size - 1
        self.text_end_idx = self.text_voc_size - 2
        self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx
        assert text_i2w[self.text_end_idx] == OUT_END_TOKEN
        assert text_i2w[self.text_begin_idx] == OUT_BEGIN_TOKEN  # 最后两个是end_idx 和begin_idx

        if self.max_train_size==-1:
            train_asts,train_texts=train_data['asts'], train_data['texts']
        else:
            train_asts, train_texts = zip(*random.sample(list(zip(train_data['asts'], train_data['texts'])),
                                                     min(self.max_train_size,
                                                         len(train_data['asts']))
                                                     )
                                      )
        train_set = self.Dataset(asts=train_asts,
                                 texts=train_texts,
                                 ast_max_size=self.ast_max_size,
                                 text_max_len=self.text_max_len,
                                 text_begin_idx=self.text_begin_idx,
                                 text_end_idx=self.text_end_idx,
                                 pad_idx=self.pad_idx)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=self.train_batch_size,
                                  shuffle=True)

        if self.warm_big_epochs is None:
            self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                                  min_rate=self.min_lr_rate,
                                  lr_decay=self.lr_decay,
                                  warm_steps=self.warm_big_epochs * len(train_loader),
                                  # max(self.max_big_epochs//10,2)*train_loader.__len__()
                                  reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()
        if self.train_mode:  # 如果进行训练
            # best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
            # self.net.load_state_dict(torch.load(best_net_path))
            # self.net.train()
            for i in range(0,self.max_big_epochs):
                # logging.info('---------Train big epoch %d/%d' % (i + 1, self.max_big_epochs))
                pbar = tqdm(train_loader)
                for j, (batch_ast_node,batch_ast_node_pos,batch_ast_edge,batch_ast_edge_pos,
                        batch_text_input,batch_text_output) in enumerate(pbar):
                    batch_ast_node=batch_ast_node.to(device)
                    batch_ast_node_pos=batch_ast_node_pos.to(device)
                    batch_ast_edge=batch_ast_edge.to(device)
                    batch_ast_edge_pos=batch_ast_edge_pos.to(device)
                    batch_text_input=batch_text_input.to(device)
                    batch_text_output=batch_text_output.to(device)
                    pred_text_output = self.net(batch_ast_node,batch_ast_node_pos,
                                                batch_ast_edge,batch_ast_edge_pos,
                                                batch_text_input)
                    loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
                    self.optimizer.zero_grad()  # 梯度置0
                    loss.backward()  # 反向传播
                    # clip_grad_norm_(self.net.parameters(),1e-2)  #减弱梯度爆炸
                    self.optimizer.step()  # 优化
                    self.scheduler.step()  # 衰减

                    # log_info = '[Big epoch:{}/{}]'.format(i + 1, self.max_big_epochs)
                    # if i+1>=self.start_valid_epoch:
                    log_info=self._get_log_fit_eval(loss=loss,
                                                    pred_tgt=pred_text_output,
                                                    gold_tgt=batch_text_output,
                                                    tgt_i2w=text_i2w
                                                    )
                    log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
                    pbar.set_description(log_info)
                    del pred_text_output,batch_ast_node,batch_ast_node_pos,batch_ast_edge,batch_ast_edge_pos,\
                        batch_text_input,batch_text_output

                del pbar
                if i+1 >= self.start_valid_epoch:
                    self.max_valid_size=len(valid_data['asts']) if self.max_valid_size==-1 else self.max_valid_size
                    valid_asts, valid_texts = zip(*random.sample(list(zip(valid_data['asts'],valid_data['texts'])),
                                                                              min(self.max_valid_size,
                                                                                  len(valid_data['asts']))
                                                                              )
                                                               )
                    worse_epochs = self._do_validation(valid_srcs=valid_asts,  # valid_data['code_asts']
                                                       valid_tgts=valid_texts,  # valid_data['texts']
                                                       tgt_i2w=text_i2w,  # valid_data['text_dic']
                                                       increase_better=True,
                                                       last=False)  # 根据验证集loss选择best_net
                    # worse_epochs=self._do_validation(valid_srcs=valid_data['asts'],
                    #                                  valid_tgts=valid_data['texts'],
                    #                                  tgt_i2w=text_i2w,
                    #                                  increase_better=True,
                    #                                  last=False)  # 根据验证集loss选择best_net
                    if worse_epochs>=self.early_stop:
                        break

        self._do_validation(valid_srcs=valid_data['asts'],
                            valid_tgts=valid_data['texts'],
                            tgt_i2w=text_i2w,
                            increase_better=True,
                            last=True)  # 根据验证集loss选择best_net

        self._logging_paramerter_num()  #需要有并行的self.net和self.model_name

    def predict(self,
                asts,
                text_i2w):
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
        self.net.eval()  # 切换测试模式
        enc=nn.DataParallel(self.net.module.enc)
        dec=nn.DataParallel(self.net.module.dec)
        # enc=BalancedDataParallel(40, self.net.module.enc.to(device), dim=0)
        # dec=BalancedDataParallel(40, self.net.module.dec.to(device), dim=0)
        # enc.eval()
        # dec.eval()
        data_set = self.Dataset(asts=asts,
                                 texts=None,
                                 ast_max_size=self.ast_max_size,
                                 text_max_len=self.text_max_len,
                                 text_begin_idx=self.text_begin_idx,
                                 text_end_idx=self.text_end_idx,
                                 pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

        data_loader = DataLoader(dataset=data_set,
                                  batch_size=self.pred_batch_size,
                                  shuffle=False)    #False!False!False!
        pred_text_id_np_batches = []  # 所有batch的预测出的id np
        with torch.no_grad():  # 取消梯度
            pbar = tqdm(data_loader)
            for batch_ast_node,batch_ast_node_pos,batch_ast_edge,batch_ast_edge_pos,batch_text_input in pbar:
                batch_ast_node = batch_ast_node.to(device)
                batch_ast_node_pos = batch_ast_node_pos.to(device)
                batch_ast_edge = batch_ast_edge.to(device)
                batch_ast_edge_pos = batch_ast_edge_pos.to(device)
                batch_text_input = batch_text_input.to(device)

                # 先跑encoder，生成编码
                batch_ast_node_enc_out,batch_ast_edge_enc_out =enc(batch_ast_node,batch_ast_node_pos,batch_ast_edge,
                                                                   batch_ast_edge_pos)

                batch_text_output: list = []  # 每步的output tensor
                if self.beam_width == 1:
                    for i in range(self.text_max_len + 1):  # 每步开启
                        pred_out = dec(ast_node_enc=batch_ast_node_enc_out,
                                       ast_edge_enc=batch_ast_edge_enc_out,
                                       text_input=batch_text_input)  # 预测该步输出 (B,text_voc_size,L_text)
                        batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  # 将该步输出加入msg output
                        if i < self.text_max_len:  # 如果没到最后，将id加入input
                            batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
                    batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.pad_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  # (B,L_tgt) 要除去pad id和begin id
                    pred_text_id_np_batches.append(batch_pred_text_np)  # [(B,L_tgt)]
                else:
                    batch_pred_text=trans_beam_search(net=dec,
                                                      beam_width=self.beam_width,
                                                      dec_input_arg_name='text_input',
                                                      length_penalty=1,
                                                      begin_idx=self.tgt_begin_idx,
                                                      pad_idx=self.pad_idx,
                                                      end_idx=self.tgt_end_idx,
                                                      ast_node_enc=batch_ast_node_enc_out,
                                                      ast_edge_enc=batch_ast_edge_enc_out,
                                                      text_input=batch_text_input
                                                      )     # (B,L_tgt)
                    pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  # [(B,L_tgt)]

        pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  # (AB,tgt_voc_size,L_tgy)
        self.net.train()  # 切换回训练模式
        # 利用字典将msg转为token
        pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_i2w, self.text_end_idx)

        return pred_texts  # 序列概率输出形状为（A,D)

    def generate_texts(self,asts,text_i2w,res_path,gold_texts,raw_data,**kwargs):
        '''
        生成src对应的tgt并保存
        :param asts:
        :param text_i2w:
        :param res_path:
        :param kwargs:
        :return:
        '''
        logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
        kwargs.setdefault('beam_width',1)
        res_dir=os.path.dirname(res_path)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        pred_texts=self.predict(asts=asts,
                                text_i2w=text_i2w
                                )
        gold_texts=self._tgt_ids2tokens(gold_texts,text_i2w,self.pad_idx)
        res_data = []
        for i,(pred_text,gold_text,raw_item) in \
                enumerate(zip(pred_texts,gold_texts,raw_data)):
            sent_bleu=self.valid_metric([pred_text],[gold_text])
            res_data.append(dict(pred_text=' '.join(pred_text),
                                 gold_text=' '.join(gold_text),
                                 sent_bleu=sent_bleu,
                                 raw_code=raw_item['code'],
                                 raw_text=raw_item['text'],
                                 ))

        with codecs.open(res_path,'w',encoding='utf-8') as f:
            json.dump(res_data,f,indent=4, ensure_ascii=False)
        self._logging_paramerter_num()  #需要有并行的self.net和self.model_name
        logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

    def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
        return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
                                                    if end_idx in code_ids else code_ids)]
                          for code_ids in code_idss]

    def _tgt_ids2tokens(self, text_id_np, text_i2w, end_idx=0, **kwargs):
        text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
                                                  if end_idx in text_ids else text_ids)]
                      for text_ids in text_id_np]
        return text_tokens

if __name__ == '__main__':

    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    model = TModel(model_dir=params['model_dir'],
                   model_name=params['model_name'],
                   model_id=params['model_id'],
                   emb_dims=params['emb_dims'],
                   ast_att_layers=params['ast_att_layers'],
                   ast_att_heads=params['ast_att_heads'],
                   ast_att_head_dims=params['ast_att_head_dims'],
                   ast_ff_hid_dims=params['ast_ff_hid_dims'],
                   text_att_layers=params['text_att_layers'],
                   text_att_heads=params['text_att_heads'],
                   text_att_head_dims=params['text_att_head_dims'],
                   text_ff_hid_dims=params['text_ff_hid_dims'],
                   drop_rate=params['drop_rate'],
                   pad_idx=params['pad_idx'],
                   train_batch_size=params['train_batch_size'],
                   pred_batch_size=params['pred_batch_size'],
                   gpu0_train_batch_size=params['gpu0_train_batch_size'],
                   max_train_size=params['max_train_size'],
                   max_valid_size=params['max_valid_size'],
                   max_big_epochs=params['max_big_epochs'],
                   regular_rate=params['regular_rate'],
                   lr_base=params['lr_base'],
                   lr_decay=params['lr_decay'],
                   min_lr_rate=params['min_lr_rate'],
                   warm_big_epochs=params['warm_big_epochs'],
                   early_stop=params['early_stop'],
                   start_valid_epoch=params['start_valid_epoch'],
                   Net=TransNet,
                   Dataset=Datasetx,
                   beam_width=params['beam_width'],
                   train_metrics=train_metrics,
                   valid_metric=valid_metric,
                   test_metrics=test_metrics,
                   train_mode=params['train_mode'])

    logging.info('Load data ...')
    # print(train_avail_data_path)
    with codecs.open(train_avail_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with codecs.open(valid_avail_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    with codecs.open(test_avail_data_path, 'rb') as f:
        test_data = pickle.load(f)

    with codecs.open(text_i2w_path, 'rb') as f:
        text_i2w = pickle.load(f)

    # with codecs.open(test_token_data_path,'r') as f:
    #     test_token_data=json.load(f)

    with codecs.open(test_raw_data_path,'r') as f:
        test_raw_data=json.load(f)

    # train_data['asts']=train_data['asts'][:1000]
    # train_data['texts']=train_data['texts'][:1000]

    test_data['asts']=test_data['asts'][:MAX_TEST_SIZE]
    test_data['texts']=test_data['texts'][:MAX_TEST_SIZE]

    # print(len(train_data['texts']), len(valid_data['texts']), len(test_data['texts']))
    model.fit(train_data=train_data,
              valid_data=valid_data,
              text_i2w=text_i2w)

    for key, value in params.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    test_eval_df=model.eval(test_srcs=test_data['asts'],
                            test_tgts=test_data['texts'],
                            tgt_i2w=text_i2w)

    logging.info('Model performance on test dataset:\n')
    for i in range(0,len(test_eval_df.columns),4):
        print(test_eval_df.iloc[:, i:i+4])

    model.generate_texts(asts=test_data['asts'],
                         text_i2w=text_i2w,
                         res_path=res_path,
                         gold_texts=test_data['texts'],
                         raw_data=test_raw_data)
