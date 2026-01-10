from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from models.common import *
import numpy as np
import os
from torch.nn.modules.loss import CrossEntropyLoss

def get_dfsp(train_dataset, config, device):
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)


    model = DFSP(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    return model,optimizer


class DFSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        

        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes), config.SA_K, context_length=self.config.context_length, fusion=self.config.fusion)
        self.weight = config.res_w


    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x, img_feature


    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature[:, 0, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        return img_feature, txt_tf

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        logits_att = torch.zeros(logits.shape[0], len(self.attributes)).cuda()
        logits_obj = torch.zeros(logits.shape[0], len(self.classes)).cuda()
        for i in range(len(self.attributes)):
            logits_att[:, i] = logits[:, np.where(att_idx==i)[0]].mean(-1)
        for i in range(len(self.classes)):
            logits_obj[:, i] = logits[:, np.where(obj_idx==i)[0]].mean(-1)        
        return logits_att, logits_obj

    def encode_text_for_open(self,idx):
        self.open_idx = idx

        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  

        return text_features,text_ft
    
    def forward_for_open(self,batch,idx,text_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]

        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768

        text_features, text_ft = text_feats[0],text_feats[1]
        
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)

        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), self.open_idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))

        
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.t()
            )   

        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, self.open_idx)

        return (logits, logits_att, logits_obj, logits_soft_prompt)



    def forward(self, batch, idx):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  
        
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)

        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))


        
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.t()
            )   

        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)

        return (logits, logits_att, logits_obj, logits_soft_prompt)

    
    def multi_attr_loss_calu(self,predict, target,pair_positive_weight=None,attr_positive_weight=None):
        loss_fn = CrossEntropyLoss()

        multi_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=attr_positive_weight)
        multi_pair_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pair_positive_weight)

        _, batch_attr, batch_obj, batch_target = target
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        batch_attr = batch_attr.float() / batch_attr.sum(dim=-1,keepdim=True)
        batch_target = batch_target.float() / batch_attr.sum(dim=-1,keepdim=True)
        
        logits, logits_att, logits_obj, logits_soft_prompt = predict



        loss_logit_df = multi_pair_loss_fn(logits, batch_target)
        loss_logit_sp = multi_pair_loss_fn(logits_soft_prompt, batch_target)



        loss_att = multi_attr_loss_fn(logits_att, batch_attr)
        loss_obj = loss_fn(logits_obj, batch_obj)

        loss = loss_logit_df + self.config.att_obj_w * (loss_att + loss_obj) + self.config.sp_w * loss_logit_sp
        return loss
    
    def single_attr_loss_calu(self,predict, target,pair_positive_weight=None,attr_positive_weight=None):
        loss_fn = CrossEntropyLoss()


        _, batch_attr, batch_obj, batch_target = target
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        
        logits, logits_att, logits_obj, logits_soft_prompt = predict



        loss_logit_df = loss_fn(logits, batch_target)
        loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
        loss_att = loss_fn(logits_att, batch_attr)
        loss_obj = loss_fn(logits_obj, batch_obj)
        loss = loss_logit_df + self.config.att_obj_w * (loss_att + loss_obj) + self.config.sp_w * loss_logit_sp
        return loss
    
    def logit_infer_multi_attr(self, predict, pairs):


        return  predict[0]


    def save_params(self,config,epoch,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        save_dict = self.state_dict()
        # save the soft embedding
        with torch.no_grad():
            if save_name == None:
                torch.save(save_dict, os.path.join(config.save_path, f"epoch_{epoch}.pt"))
            else:
                torch.save(save_dict, os.path.join(config.save_path,save_name))

    # def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    #     self.load_state_dict(state_dict,strict=strict,assign=assign)
    #     return 
    