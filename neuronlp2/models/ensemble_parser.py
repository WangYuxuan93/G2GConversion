# coding:utf-8
# @Time     : 2021/10/11 10:31 AM
# @Author   : jeffrey
import os
import json
from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.io import get_logger
from neuronlp2.models.sdp_biaffine_parser import SDPBiaffineParser


class EnsembleParser(nn.Module):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels, device=torch.device('cpu'), model_type="Biaffine", embedd_word=None, embedd_char=None, embedd_pos=None,
                 use_pretrained_static=True, use_random_static=False, use_elmo=False, elmo_path=None, pretrained_lm='none', lm_path=None, num_lans=1, model_paths=None, merge_by='logits', beam=5,
                 old_label=0,lm_config=None):
        super(EnsembleParser, self).__init__()

        self.pretrained_lm = pretrained_lm
        self.merge_by = merge_by
        self.networks = []
        self.use_pretrained_static = use_pretrained_static
        self.use_random_static = use_random_static
        self.model_type = model_type
        self.beam = beam
        assert merge_by in ['logits', 'probs']
        logger = get_logger("Ensemble")
        logger.info("Number of models: %d (merge by: %s)" % (len(model_paths), merge_by))
        if model_type == "Biaffine":
            for i, path in enumerate(model_paths):
                model_name = os.path.join(path, 'model.pt')
                logger.info("Loading sub-model from: %s" % model_name)
                hyp = hyps[i]
                network = SDPBiaffineParser(hyp, num_pretrained[i], num_words[i], num_chars[i], num_pos[i], num_labels[i], device=device, pretrained_lm=pretrained_lm, lm_path=lm_path,
                                         use_pretrained_static=use_pretrained_static, use_random_static=use_random_static, use_elmo=use_elmo, elmo_path=elmo_path, num_lans=num_lans,
                                         log_name='Network-' + str(len(self.networks)), method=hyp["g2gtype"], old_label=old_label,lm_config=lm_config)
                network = network.to(device)
                network.load_state_dict(torch.load(model_name, map_location=device)["state_dict"], strict=False)
                self.networks.append(network)
        else:
            print("Ensembling %s not supported." % model_type)
            exit()
        self.hyps = self.networks[0].hyps
        self.use_elmo = any([network.use_elmo for network in self.networks])
        # has_roberta = any([network.pretrained_lm == "roberta" for network in self.networks])
        # if has_roberta:
        #    self.pretrained_lm = "roberta"
        # else:
        #    self.pretrained_lm = pretrained_lm
        self.pretrained_lms = [network.pretrained_lm for network in self.networks]
        self.lm_paths = [network.lm_path for network in self.networks]
        self.lan_emb_as_input = False

    def eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()


    def decode(self, input_words, input_pretrained, input_chars, input_poss, mask=None, bpes=None, first_idx=None, target_mask=None,input_elmo=None, lan_id=None, leading_symbolic=0, beam=5,
               src_heads=None, src_types=None,_src_heads=None,_src_types=None,method=None):
        if self.model_type == "Biaffine":
            if self.merge_by == 'logits':
                mask_3D= None
                arc_logits_list, rel_logits_list = [], []
                for i, network in enumerate(self.networks):
                    input_word, input_char, input_pos = input_words[i], input_chars[i], input_poss[i]
                    sub_bpes, sub_first_idx = bpes, first_idx
                    arc_logits, rel_logits,mask_3D = network.get_logits(input_word, input_pretrained, input_char, input_pos, mask=mask, bpes=sub_bpes, first_idx=sub_first_idx, input_elmo=input_elmo,
                                                                lan_id=lan_id, leading_symbolic=leading_symbolic)
                    arc_logits_list.append(arc_logits)
                    rel_logits_list.append(rel_logits)
                arc_logits = sum(arc_logits_list)
                rel_logits = sum(rel_logits_list)


            # arc_preds0 = arc_logits_list[0].ge(0)  # 多个arc
            # arc_preds1 = arc_logits_list[1].ge(0)
            # arc_preds = arc_preds0 | arc_preds1
            arc_preds = arc_logits.ge(0)
            # (batch_size, len_c, len_h, n_rels)
            # transposed_type_logits = rel_logits.permute(0, 2, 3, 1)  # permute重新排列张量
            # (batch_size, seq_len, seq_len)
            # 没做softmax
            # type_preds = transposed_type_logits.argmax(-1)  # 在只有一个label的情况下找到最大的索引
            type_preds = F.softmax(rel_logits, dim=-1)
            # jeffrey: 加一个mask 过滤不在目标标签体系中的label:

            # logger.info(type_preds[0, 1, 2, :])
            type_preds = type_preds.argmax(dim=-1)


            return arc_preds.float()*mask_3D,type_preds
        else:
            print("Ensembling %s not supported." % self.model_type)
            exit()