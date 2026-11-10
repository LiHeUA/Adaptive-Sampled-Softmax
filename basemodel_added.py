import logging, os
from typing import Union, Dict, Tuple, List
from torch import Tensor

# HL
import numpy as np
# HL end

import torch
from torch import optim
import pytorch_lightning
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .scorer import InnerProductScorer
from .loss_func import FullSoftmax, SampledSoftmax
from .utils import color_dict
from .sampler import (UniformSampler, PopularSampler, 
                      MIDXSamplerUniform, MIDXSamplerUniLarge, MIDXSamplerPop, MIDXSamplerPopLarge,
                      SphereSampler, RFFSampler, DynamicSampler,
                      SphereSamplerAppr, RffSamplerAppr)

from .init import normal_initialization

import time

# HL
#from torch.utils.tensorboard import SummaryWriter
# HL end

class BaseModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('MIDX')
        parent_parser.add_argument("--learning_rate", type=float, default=0.001, help='learning rate')
        parent_parser.add_argument("--learner", type=str, default="adam", help='optimization algorithm')
        parent_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay coefficient')
        parent_parser.add_argument('--epochs', type=int, default=50, help='training epochs')
        parent_parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parent_parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size')
        parent_parser.add_argument('--val_n_epoch', type=int, default=1, help='valid epoch interval')
        parent_parser.add_argument('--early_stop_patience', type=int, default=10, help='early stop patience')
        parent_parser.add_argument('--gpu', type=int, action='append', default=None, help='gpu number')
        parent_parser.add_argument('--init_method', type=str, default='xavier_normal', help='init method for model')
        parent_parser.add_argument('--init_range', type=float, help='init range for some methods like normal')
        parent_parser.add_argument('--sampler', type=str, default=None, help='which sampler to use')
        parent_parser.add_argument('--num_cluster', type=int, default=16, help='number of codewords for midx-based samplers')
        parent_parser.add_argument('--num_neg', type=int, default=50, help='the number of negative samples')
        # Added
        parent_parser.add_argument('--target_ess', type=int, default=50, help='the target ess, add more neg samples until ess is larger than this threshold')
        parent_parser.add_argument('--num_neg_step', type=int, default=50, help='add more neg samples with this step')
        parent_parser.add_argument('--ratio_ess_cv', type=float, default='0.9', help='threhold ratio of cv or ess')
        # Added End
        return parent_parser


    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.target_vector = None
        if self.config['monitor_metric'] is None:
            self.val_metric = 'train_loss'
        else:
            self.val_metric = self.config['monitor_metric']
        self.num_items = train_data.num_items
        self.item_freq = train_data.item_freq
        self.score_fn = InnerProductScorer()

        self.sampler = self.configure_sampler()
        self.loss_fn = self.configure_loss()
        
        self.console_logger = logging.getLogger('MIDX')
        self.console_logger.info(f"Number of Items: {self.num_items}")

        if hasattr(train_data, "num_feat"):
            self.console_logger.info(f"Number of Feat: {train_data.num_feat}")
            self.num_feat = train_data.num_feat # work for extreme classification task
        
        # Added
        self.ess = 0
        self.cv = 0
        self.itrcnt = 0
       # Added End
     

    def forward(self, batch, pad2inf=True):
        output = {}
        query = self.construct_query(batch)
  
        pos_item = batch['target']
        pos_vec = self.encode_target(pos_item)
        pos_score = self.score_fn(query, pos_vec)
        if pad2inf:
            pos_score[batch['target']==0] = -float('inf')
        output['pos_score'] = pos_score

        if self.sampler is not None:    # sampled softmax
            log_pos_prob, neg_id, log_neg_prob = self.sampling(query, self.config['num_neg'], pos_item)
            neg_vec = self.encode_target(neg_id)
            output['neg_score'] = self.score_fn(query, neg_vec)
            output['log_pos_prob'] = log_pos_prob.detach()
            output['log_neg_prob'] = log_neg_prob.detach()
        else: # full softmax
            output['full_score'] = self.score_fn(query, self.item_vector)
            
        # Added
        with torch.no_grad():
            neg_score = output['neg_score']
            num_ng = self.config['num_neg']
            zk = torch.exp(neg_score)/num_ng
            zk.clamp_(min=0.0, max=1e9)
            zk = zk/torch.exp(log_neg_prob)
            
            denominator = torch.sum(zk**2, dim=-1)
            denominator.add_(1e-9)
            ess = torch.sum(zk, dim=-1)**2 / denominator
            ess.clamp_(min=1e-5, max=num_ng)
            cv = 1.0/ess
            cv = cv.add(-1.0/num_ng)

            ess = torch.mean(ess)
            cv = torch.mean(cv)
            
            self.ess += ess
            self.cv += cv
            self.itrcnt += 1
        # Added End
        
        return output        

    def on_train_epoch_start(self) -> None:
        if self.sampler is not None:
            self.sampler.update(self.item_vector)
            
        # Added
        with torch.no_grad():
            self.ess = 0
            self.cv = 0
            self.itrcnt = 0
            self.time_epoch_start = time.time()
        # Added End

    def training_epoch_end(self, outputs):  
        loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
        self.log_dict(loss_metric)
        output_dict = self.trainer.logged_metrics
        output_dict.update({'epoch': self.trainer.current_epoch})
        self.console_logger.info(color_dict(output_dict, False))
        if self.config['mode'] == 'tune':
            metric = {}
            for k, v in output_dict.items():
                if isinstance(v, torch.Tensor):
                    metric[k] = v.item()
                else:
                    metric[k] = v
            metric['default'] = metric[self.config['monitor_metric']]
            # nni.report_intermediate_result(metric)
        
        # Added
        with torch.no_grad():
            ess = self.ess/num_iter
            num_ng = self.config['num_neg']
            cv = 1.0/ess
            cv = cv.add(-1.0/num_ng)
            
            current_epoch = self.current_epoch

            if self.config['ess_cv_fix'] == 'ess':
                if self.config['target_ess']>0: # using a pre-defined ess
                    if ess<self.config['target_ess']:
                        self.config['num_neg'] += self.config['num_neg_step']
                else:   # if ess=-1, then use adaptive ess and
                        # take the avg ess of the leading 3 epochs
                        # as the baseline, and requires .9*baseline
                    if self.current_epoch==3:   
                        self.config['target_ess'] = torch.mean(self.ess_stored[0:2])*self.config['ratio_ess_cv']
            elif self.config['ess_cv_fix'] == 'cv':
                if self.config['target_cv']>0: # using a pre-defined cv threshold
                    if cv>self.config['target_cv']:
                        self.config['num_neg'] += self.config['num_neg_step']
                        self.config['target_cv'] = cv/self.config['ratio_ess_cv']
                else:   # if cv=-1, then use adaptive cv and
                        # take the avg cv of the leading 3 epochs
                        # as the baseline, and requires .9*baseline
                    if self.current_epoch==3:   
                        self.config['target_cv'] = torch.mean(self.cv_stored[0:2])/self.config['ratio_ess_cv']
            # else do nothing
            
        # Added End