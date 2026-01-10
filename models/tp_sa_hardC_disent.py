## Remove this later
import argparse
import os

from typing import List
import clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from models.common import  CrossResidualAttentionBlock,ResidualAttentionBlock
from utils.loss import *
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_tp_sa_hardC_disent(train_dataset, config, device, prompt_template="a photo of x x"):
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    # text --> token idx
    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )

    # token idx --> token embedding
    origin_token_embedding = clip_model.token_embedding(tokenized.to(device))


    # get the token embedding of all attr / object
    offset = len(attributes)
    offset2 = len(attributes) + len(classes)
    with torch.no_grad():
        primitive_embedding = torch.zeros(
            (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
        )

        for idx, rep in enumerate(origin_token_embedding[0:offset2]):
            eos_idx = tokenized[idx].argmax()
            primitive_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    
        # get the token embedding of 'object'
        tokenized_o = clip.tokenize('object',context_length=config.context_length)
        embedding_o = clip_model.token_embedding(tokenized_o.to(device))
        eos_idx = tokenized_o[0].argmax()
        embedding_o = torch.mean(embedding_o[0,1:eos_idx,:],axis=0)

    


    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init,
                           context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)

    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
    ctx_vectors = ctx_vectors.repeat(3,1,1)

    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)





    soft_embedding = nn.Parameter(ctx_vectors.to(device)) 
    primitive_embedding = nn.Parameter(primitive_embedding.to(device))


    encoder_width = soft_embedding.shape[-1]
    image_text_encoder_layer = nn.TransformerEncoderLayer(d_model = encoder_width,nhead=config.num_heads,batch_first = True)
    image_text_encoder = nn.TransformerEncoder(encoder_layer=image_text_encoder_layer, num_layers=config.num_encoder_layers).to(device)


    tpsa = TPSA_hardC_Disent(
        clip_model,
        config,
        offset,
        soft_embedding,
        primitive_embedding,
        embedding_o,
        token_ids,
        device=device,
        enable_pos_emb=True,
        image_text_encoder = image_text_encoder,
        #attr_dropout=config.attr_dropout
    )

    tpsa = tpsa.to(device)


    if config.train_vocab:
        print("Training vocabulary embeddings")
        optimizer = torch.optim.Adam(
                [
                    {'params':tpsa.soft_embeddings},
                    {'params':tpsa.image_text_encoder.parameters()},
                    {'params':tpsa.primitive_embeddings}
                ], 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
    else:
        optimizer = torch.optim.Adam(
                [
                    {'params':tpsa.soft_embeddings},
                    {'params':tpsa.image_text_encoder.parameters()},
                ], 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
        

    optimizer.add_param_group(
        {
            'params': tpsa.attr_disentangler.parameters(),
            'lr': config.lr,
            'weight_decay': config.weight_decay
        }
    )
    optimizer.add_param_group(
        {
            'params': tpsa.obj_disentangler.parameters(),
            'lr': config.lr,
            'weight_decay': config.weight_decay
        }
    )


    return tpsa, optimizer





class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

class TPSA_hardC_Disent(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: List[torch.nn.Parameter],
        primitive_embeddings: torch.nn.Parameter,
        object_word_embedding: torch.nn.Parameter,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
        image_text_encoder = None,
        attr_dropout=0.3,
    ):
        '''
            initial Two prompt + self attention
            Args:
                token_ids : the clip token ids of 'a photo of x x'
                word_embedding: the clip token embedding of word 'object'
        '''
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        output_dim = self.clip_model.visual.output_dim
        self.primitive_embeddings = primitive_embeddings

        self.offset = offset
        self.primitive_num = self.primitive_embeddings.shape[0]

        self.object_word_embedding = object_word_embedding

        self.image_text_encoder = image_text_encoder

        #self.query_token = nn.Parameter(torch.randn([1,self.frozen_attr_embeddings.shape[-1]])).to(self.device)
        #self.query_token.requires_grad = True
        #self.cls_layer = nn.Parameter(torch.randn())
        self.multi_label_loss = config.multi_label_loss

        self.train_vocab = config.train_vocab

        self.attr_dropout = nn.Dropout(attr_dropout)

        # self.patch_norm = nn.LayerNorm(output_dim)
        self.attr_disentangler = Disentangler(primitive_embeddings.shape[-1])
        self.obj_disentangler = Disentangler(primitive_embeddings.shape[-1])


        if isinstance(config.primitive_topk,list):
            self.attr_topk = config.primitive_topk[0]    # set 10
            self.obj_topk = config.primitive_topk[1] 
        else:
            self.attr_topk =  config.primitive_topk
            self.obj_topk = config.primitive_topk

        print('select topk primtive:', config.primitive_topk)

        self.pair_to_attr_mask = None
        self.pair_to_obj_mask = None
        
        if hasattr(config, 'encoded_comp') and config.encoded_comp:
            self.encoded_comp = True
        else:
            self.encoded_comp = False

        self.traing_test_flag = False

    def construct_token_tensors(self):

        # repeat 'a photo of x x' for s+o+c times
        # class_token_ids = self.token_ids.repeat(self.primitive_num + len(pair_idx), 1)
        class_token_ids = self.token_ids.repeat(self.primitive_num, 1)

        # convert token ids to token embedding
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)
        
        eos_idx = int(self.token_ids[0].argmax())

        eos_token_embedding = token_tensor[0][eos_idx]
        none_token_embedding = token_tensor[0][-1]

        
        if self.train_vocab:
            primitive_embeddings = self.attr_dropout(self.primitive_embeddings)
        else:
            primitive_embeddings = self.primitive_embeddings

        # token embedding for attrs :  [sot][x1][x2][x3][s]['object'][eot]
        token_tensor[0:self.offset, eos_idx - 2, :] = primitive_embeddings[0:self.offset].type(self.clip_model.dtype)
        token_tensor[0:self.offset, eos_idx - 1, :] = self.object_word_embedding.type(self.clip_model.dtype)

        # token embedding for objs: [sot][x1][x2][x3][o][eot][none]
        token_tensor[self.offset:self.primitive_num , eos_idx - 2, :] = primitive_embeddings[self.offset:].type(self.clip_model.dtype)
        token_tensor[self.offset:self.primitive_num , eos_idx - 1, :] = eos_token_embedding
        token_tensor[self.offset:self.primitive_num , eos_idx , :] = none_token_embedding


        # construct the prompt for composition: [sot][x1][x2][x3][s][o][eot]
        # attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        # token_tensor[self.primitive_num: , eos_idx - 2, :] = primitive_embeddings[0:self.offset][
        #     attr_idx
        # ].type(self.clip_model.dtype)

        # token_tensor[self.primitive_num:, eos_idx - 1, :] = primitive_embeddings[self.offset:][
        #     obj_idx
        # ].type(self.clip_model.dtype)


        # add learnable prompt embedding: s/o/c
        len_soft_embedding = len(self.soft_embeddings[0])
        token_tensor[
            0:self.offset, 1 : len_soft_embedding + 1, :
        ] = self.soft_embeddings[0].type(self.clip_model.dtype).squeeze(0)
        

        token_tensor[
            self.offset:self.primitive_num, 1 : len_soft_embedding + 1, :
        ] = self.soft_embeddings[1].type(self.clip_model.dtype).squeeze(0)


        # token_tensor[
        #     self.primitive_num: , 1 : len_soft_embedding + 1, :
        # ] = self.soft_embeddings[2].type(self.clip_model.dtype).squeeze(0)

        return token_tensor

    def construct_aux_pair_token_tensors(self, aux_attr_idx, aux_obj_idx):
        """
        构造 attribute 与多个 object 组合的 token embedding。
        aux_obj_idx: Tensor of shape [B, topk]，每个样本的 topk object 索引
        输出: Tensor of shape [B, A * topk, context_len, embed_dim]
        """
        B, Ka = aux_attr_idx.shape
        B, Ko = aux_obj_idx.shape

        # Step 1: 生成每个样本的 K*K 个组合 pair_idx: [B*K*K, 2]
        grid_attr, grid_obj = torch.meshgrid(
            torch.arange(Ka, device=aux_attr_idx.device),
            torch.arange(Ko, device=aux_obj_idx.device),
            indexing='ij'
        )  # [K, K]
        grid_attr = grid_attr.flatten()   # [K*K]
        grid_obj = grid_obj.flatten()     # [K*K]

        # 组合为每个样本的pair
        all_attr = aux_attr_idx[:, grid_attr]  # [B, K*K]
        all_obj  = aux_obj_idx[:, grid_obj]    # [B, K*K]

        attr_idx = all_attr.flatten()  # [B*K*K]
        obj_idx  = all_obj.flatten()   # [B*K*K]

        # Step 2: 拿 token template [ctx_len] 并复制成 [B*K*K, ctx_len]
        pair_num = attr_idx.shape[0]
        class_token_ids = self.token_ids.repeat(pair_num, 1)  # [B*K*K, ctx_len]
        
        # Step 3: lookup token embedding
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)  # [B*K*K, ctx_len, D]

        eos_idx = int(self.token_ids[0].argmax())  # assume same for all

        # Step 4: 替换 soft tokens
        if self.train_vocab:
            primitive_embeddings = self.attr_dropout(self.primitive_embeddings)
        else:
            primitive_embeddings = self.primitive_embeddings
        token_tensor[:, eos_idx - 2, :] = primitive_embeddings[attr_idx].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = primitive_embeddings[obj_idx + self.offset].type(self.clip_model.dtype)

        len_soft_embedding = len(self.soft_embeddings[2])
        token_tensor[
              : , 1 : len_soft_embedding + 1, :
        ] = self.soft_embeddings[2].type(self.clip_model.dtype).squeeze(0)

        token_tensor = token_tensor.view(B, Ka*Ko, token_tensor.shape[1], token_tensor.shape[2])


        return token_tensor
        


    def encode_image(self, imgs,return_patch=True):
        return self.clip_model.encode_image(imgs,return_patch=return_patch)
    


    


    def construct_dynamic_gt(self, topk_attr_idx, topk_obj_idx, attr_gt, obj_gt):
        """
        生成动态组合空间的ground truth标签。
        aux_attr_idx, aux_obj_idx: [B, K]
        attr_gt: [B, A] multi-hot
        obj_gt:  [B, O] multi-hot

        返回: gt_comp [B, K*K]，每个位置表示该组合是否为GT
        """
        B, Ka = topk_attr_idx.shape
        B, Ko = topk_obj_idx.shape

        # Step 1: 获取每个样本的 K*K 个组合（用 meshgrid）
        grid_attr, grid_obj = torch.meshgrid(
            torch.arange(Ka, device=topk_attr_idx.device),
            torch.arange(Ko, device=topk_attr_idx.device),
            indexing='ij'
        )  # shape: [K, K]
        grid_attr = grid_attr.flatten()  # [K*K]
        grid_obj = grid_obj.flatten()    # [K*K]

        all_attr = topk_attr_idx[:, grid_attr]  # [B, K*K]
        all_obj  = topk_obj_idx[:, grid_obj]    # [B, K*K]

        # Step 2: 对应位置是否为 GT？
        # attr_gt: [B, A], all_attr: [B, K*K] ∈ [0, A)
        # Gather对应的GT值: [B, K*K]
        attr_mask = torch.gather(attr_gt, 1, all_attr)  # [B, K*K]
        obj_mask  = torch.gather(obj_gt, 1, all_obj)    # [B, K*K]

        gt_comp = (attr_mask * obj_mask)  # [B, K*K] ∈ {0, 1}

        return gt_comp, all_attr, all_obj
    
    def random_shuffle_dix(self,idx):
        B,k = idx.shape
        rand_idx = torch.rand(B, k).argsort(dim=1).to(idx.device)  # 每行打乱
        topk_indices_shuffled = torch.gather(idx, 1, rand_idx)

        return topk_indices_shuffled
    
    # 由于推理时是logits加和，设置过小会导致完全取决于这个辅助的组合分支的预测
    # 推理算法可以优化
    def map_comp_logits_to_C(self,comp_logits, comp_attr_idx, comp_obj_idx, pair_idx, default_val=-0.0):
        """
        将组合预测结果 comp_logits 映射回 C 个固定的类别。

        comp_logits:     [B, N]
        comp_attr_idx:   [B, N]
        comp_obj_idx:    [B, N]
        composition_set: List[(attr_idx, obj_idx)] 长度为 C
        返回: final_logits [B, C]
        """
        B, N = comp_logits.shape
        C = len(pair_idx)

        # 1. 构造 composition_set 的映射表
        comp2idx = { (int(a), int(o)): i for i, (a, o) in enumerate(pair_idx) }


        # 2. 初始化结果
        final_logits = torch.full((B, C), default_val, device=comp_logits.device, dtype=comp_logits.dtype)

        # 3. 遍历每个样本，将匹配到的位置放入 final_logits 中
        for b in range(B):
            for n in range(N):
                key = (int(comp_attr_idx[b, n].item()), int(comp_obj_idx[b, n].item()))
                if key in comp2idx:
                    c_idx = comp2idx[key]
                    final_logits[b, c_idx] = comp_logits[b, n]
        

        return final_logits

    def logit_infer_multi_attr(self, predict, pairs):

        if self.config.aux_loss:
            # if aux loss is used, the predict will contain aux_attr_logits and aux_obj_logits
            attr_logits, obj_logits, comp_logits,comp_gt, aux_attr_logits, aux_obj_logits,attr_idx,obj_idx = predict
            attr_logits = (attr_logits + aux_attr_logits)/2
            obj_logits = (obj_logits + aux_obj_logits)/2
        else:
            attr_logits, obj_logits,comp_logits,comp_gt,attr_idx,obj_idx = predict


        comp_logits = self.map_comp_logits_to_C(comp_logits,attr_idx,obj_idx,pairs, default_val=comp_logits.mean())

        


        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_logits = 0 if self.config.attr_inference_weight == 0 else attr_logits[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_logits = 0 if self.config.obj_inference_weight == 0 else obj_logits[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            

            comp_logits[:, i_comp] +=  weighted_attr_logits + weighted_obj_logits

        return comp_logits




    def forward(self, batch_img, idx=None,text_features=None):

        
        if isinstance(batch_img,list):
            attr_gt, obj_gt, pair_gt = batch_img[1].to(self.device),batch_img[2].to(self.device),batch_img[3].to(self.device)
            batch_img = batch_img[0].to(self.device)

            if hasattr(self.config,'data_type') and self.config.data_type == 'single':
                attr_gt = F.one_hot(attr_gt,num_classes= self.offset)
            obj_gt = F.one_hot(obj_gt,num_classes=self.primitive_num - self.offset)

        else:
            batch_img = batch_img.to(self.device)

        

        # extract 
        batch = batch_img.shape[0]
        batch_img,patch_features = self.encode_image(batch_img,return_patch=True)


        if text_features == None:
            token_tensors = self.construct_token_tensors()

            text_features= self.text_encoder(
                self.token_ids,
                token_tensors,
                enable_pos_emb=self.enable_pos_emb,
                return_token = True
            )

        text_features = text_features.float()
        batch_img, patch_features = batch_img.float(), patch_features.float()
        

        num_image_token = patch_features.shape[1]

        
        primitive_text_features = text_features[:self.primitive_num,:]


        # Encoded primitive features and img features
        batch_img_attr = self.attr_disentangler(batch_img)
        batch_img_obj = self.obj_disentangler(batch_img)

        encoder_input = torch.cat( 
                [
                batch_img_attr.unsqueeze(1),
                batch_img_obj.unsqueeze(1),
                patch_features, 
                primitive_text_features.repeat(batch,1,1)] 
            ,dim=1)
        
        img_text_features = self.image_text_encoder(encoder_input)
        
        cls_attr,cls_obj,patch_fs,encoded_primitive_fs = torch.split(img_text_features,[1,1,num_image_token,self.primitive_num],dim=1)
    
        # get logits
        cls_attr,cls_obj =cls_attr.squeeze(1),cls_obj.squeeze(1)
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in [cls_attr,cls_obj]]
        cls_attr,cls_obj = normalized_img_features

        # normlize before calculating logits
        primitive_text_features = primitive_text_features / primitive_text_features.norm(dim=-1, keepdim=True)
        
        
        attr_logits = (
            self.clip_model.logit_scale.exp()
            * cls_attr
            @ primitive_text_features[0:self.offset].t()
        )

        obj_logits = (
            self.clip_model.logit_scale.exp()
            * cls_obj
            @ primitive_text_features[self.offset:self.primitive_num].t()
        )


        # aux loss for direction; can only use for 3 branch
        # 保证GT被选取
        if self.training or self.traing_test_flag:
            aux_attr_logits = attr_logits + (attr_gt * self.clip_model.logit_scale.exp())
            aux_obj_logits =  obj_logits   +   (obj_gt    *  self.clip_model.logit_scale.exp())
        else:
            aux_attr_logits = attr_logits
            aux_obj_logits = obj_logits

        # get random shuffled topk index
        topk_attr_idx = self.random_shuffle_dix(torch.topk(aux_attr_logits, self.attr_topk, dim=1).indices)
        topk_obj_idx =  self.random_shuffle_dix(torch.topk(aux_obj_logits,  self.obj_topk , dim=1).indices)
        

        comp_token_tensor = self.construct_aux_pair_token_tensors(topk_attr_idx,topk_obj_idx)

        comp_text_features= self.text_encoder(
                self.token_ids,
                comp_token_tensor,
                enable_pos_emb=self.enable_pos_emb,
                return_token = True
            )
        
        if self.encoded_comp:

            num_comp = comp_text_features.shape[1]
            encoder_input = torch.cat( 
                [
                batch_img.unsqueeze(1),
                patch_features, 
                comp_text_features] 
            ,dim=1)
        
            img_text_features = self.image_text_encoder(encoder_input)
            
            cls_comp,patch_fs,encoded_primitive_fs = torch.split(img_text_features,[1,num_image_token,num_comp],dim=1)
            cls_comp = cls_comp.squeeze(1)
        

        comp_text_features = comp_text_features.float()
        comp_text_features = comp_text_features / comp_text_features.norm(dim=-1,keepdim=True)

        
        if self.encoded_comp:
            cls_comp = cls_comp / cls_comp.norm(dim=-1,keepdim=True)
            comp_logits = torch.einsum(
                        "bd, bkd->bk", 
                        cls_comp, 
                        comp_text_features * self.clip_model.logit_scale.exp()
                )
        else:
            batch_img = batch_img / batch_img.norm(dim=-1,keepdim=True)
            comp_logits = torch.einsum(
                        "bd, bkd->bk", 
                        batch_img, 
                        comp_text_features * self.clip_model.logit_scale.exp()
                )

        dynamic_gt, attr_idx, obj_idx = self.construct_dynamic_gt(topk_attr_idx,topk_obj_idx,attr_gt,obj_gt)


        if self.config.aux_loss:
            return [attr_logits,obj_logits,comp_logits,dynamic_gt, aux_attr_logits,aux_obj_logits,attr_idx, obj_idx]
        else:
            return [attr_logits,obj_logits,comp_logits,dynamic_gt,attr_idx, obj_idx]


    def multi_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
        _, batch_attr, batch_obj, batch_target = target

        if self.config.aux_loss:
            # if aux loss is used, the predict will contain aux_attr_logits and aux_obj_logits
            attr_logits, obj_logits, comp_logits,comp_target, aux_attr_logits, aux_obj_logits,_,_ = predict
        else:
            attr_logits, obj_logits,comp_logits,comp_target,_,_ = predict

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        comp_target = comp_target.cuda()
        loss_fn = nn.CrossEntropyLoss()


        if self.multi_label_loss == 'ASL':
            multi_attr_loss_fn = AsymmetricLossOptimizedMean()
        elif self.multi_label_loss == 'CrossEntropy':
            multi_attr_loss_fn = nn.CrossEntropyLoss()
            multi_pair_loss_fn = nn.CrossEntropyLoss()
            batch_attr = batch_attr.float() / batch_attr.sum(dim=-1,keepdim=True)
            comp_target = comp_target.float() / comp_target.sum(dim=-1,keepdim=True)
        elif self.multi_label_loss == 'onewayLogSumExp':
            multi_attr_loss_fn = OneWayLogSumExpLoss()
            multi_pair_loss_fn = OneWayLogSumExpLoss()
        elif self.multi_label_loss == 'twowayLogSumExp':
            multi_attr_loss_fn = TwoWayLogSumExp()
            multi_pair_loss_fn = TwoWayLogSumExp()
        elif  self.multi_label_loss == 'MultiLabelSoftMaxOut':
            multi_attr_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
            multi_pair_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
        elif self.multi_label_loss == 'samplewayLogSumExp':
            multi_attr_loss_fn = SampleWayLogSumExpLoss()
            multi_pair_loss_fn = SampleWayLogSumExpLoss()
        elif self.multi_label_loss == "BCE":
            multi_attr_loss_fn = nn.BCEWithLogitsLoss()
            multi_pair_loss_fn = nn.BCEWithLogitsLoss()


        loss_attr = multi_attr_loss_fn(attr_logits, batch_attr)

        loss_obj = loss_fn(obj_logits, batch_obj)

        loss_comp = multi_pair_loss_fn(comp_logits, comp_target)

        loss = loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight +\
                loss_comp     

        if self.config.aux_loss:
            loss_attr_aux = multi_attr_loss_fn(aux_attr_logits, batch_attr)
            loss_obj_aux = loss_fn(aux_obj_logits, batch_obj)
        
            loss += loss_attr_aux  +\
                    loss_obj_aux 

        
        return loss


    # def multi_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
    #     _, batch_attr, batch_obj, batch_target = target

    #     if self.config.aux_loss:
    #         # if aux loss is used, the predict will contain aux_attr_logits and aux_obj_logits
    #         attr_logits, obj_logits, comp_logits,comp_target, aux_attr_logits, aux_obj_logits,_,_ = predict
    #     else:
    #         attr_logits, obj_logits,comp_logits,comp_target,_,_ = predict

    #     batch_attr = batch_attr.cuda()
    #     batch_obj = batch_obj.cuda()
    #     batch_target = batch_target.cuda()
    #     loss_fn = nn.CrossEntropyLoss()


    #     if self.multi_label_loss == 'ASL':
    #         multi_attr_loss_fn = AsymmetricLossOptimizedMean()
    #     elif self.multi_label_loss == 'CrossEntropy':
    #         multi_attr_loss_fn = nn.CrossEntropyLoss()
    #         multi_pair_loss_fn = nn.CrossEntropyLoss()
    #         batch_attr = batch_attr.float() / batch_attr.sum(dim=-1,keepdim=True)
    #         batch_target = batch_target.float() / batch_attr.sum(dim=-1,keepdim=True)
    #     elif self.multi_label_loss == 'onewayLogSumExp':
    #         multi_attr_loss_fn = OneWayLogSumExpLoss()
    #         multi_pair_loss_fn = OneWayLogSumExpLoss()
    #     elif self.multi_label_loss == 'twowayLogSumExp':
    #         multi_attr_loss_fn = TwoWayLogSumExp()
    #         multi_pair_loss_fn = TwoWayLogSumExp()
    #     elif  self.multi_label_loss == 'MultiLabelSoftMaxOut':
    #         multi_attr_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
    #         multi_pair_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
    #     elif self.multi_label_loss == 'samplewayLogSumExp':
    #         multi_attr_loss_fn = SampleWayLogSumExpLoss()
    #         multi_pair_loss_fn = SampleWayLogSumExpLoss()
    #     elif self.multi_label_loss == "BCE":
    #         multi_attr_loss_fn = nn.BCEWithLogitsLoss()
    #         multi_pair_loss_fn = nn.BCEWithLogitsLoss()


    #     loss_attr = multi_attr_loss_fn(attr_logits, batch_attr)

    #     loss_obj = loss_fn(obj_logits, batch_obj)

    #     loss_comp = multi_pair_loss_fn(comp_logits, comp_target)

    #     loss = loss_attr * self.config.attr_loss_weight +\
    #            loss_obj * self.config.obj_loss_weight +\
    #             loss_comp     

    #     if self.config.aux_loss:
    #         loss_attr_aux = multi_attr_loss_fn(aux_attr_logits, batch_attr)
    #         loss_obj_aux = loss_fn(aux_obj_logits, batch_obj)
        
    #         loss += loss_attr_aux  +\
    #                 loss_obj_aux 

        
    #     return loss
    
    def single_attr_loss_calu(self, predict, target):
        
        _, batch_attr, batch_obj, batch_target = target

        if self.config.aux_loss:
            # if aux loss is used, the predict will contain aux_attr_logits and aux_obj_logits
            attr_logits, obj_logits, comp_logits,comp_target, aux_attr_logits, aux_obj_logits,_,_ = predict
        else:
            attr_logits, obj_logits,comp_logits,comp_target,_,_ = predict

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        comp_target = comp_target.cuda()
        comp_target = comp_target.argmax(dim=1)


        loss_fn = nn.CrossEntropyLoss()



        loss_attr = loss_fn(attr_logits, batch_attr)

        loss_obj = loss_fn(obj_logits, batch_obj)

        loss_comp = loss_fn(comp_logits, comp_target)

        loss = loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight +\
                loss_comp     

        if self.config.aux_loss:
            loss_attr_aux = loss_fn(aux_attr_logits, batch_attr)
            loss_obj_aux = loss_fn(aux_obj_logits, batch_obj)
        
            loss += loss_attr_aux  +\
                    loss_obj_aux 

        
        return loss

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors()

        batch_size = 1024
        text_features = torch.Tensor().to(self.device).type(self.clip_model.dtype)
        for i in range(0, len(token_tensors), batch_size):
            batch_token_tensors = token_tensors[i:i+batch_size]
            batch_text_features = self.text_encoder(
            self.token_ids,
            batch_token_tensors,
            enable_pos_emb=self.enable_pos_emb,
            )
            
            text_features = torch.cat([text_features,batch_text_features],dim=0)

        return text_features

    
    def forward_for_open(self, batch,idx, text_feats):
        return self.forward(batch,idx,text_features=None)


    def save_params(self,config,epoch,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        save_dict = {"soft_embeddings":self.soft_embeddings,
                     "image_text_encoder":self.image_text_encoder.state_dict(),
                     "primitive_embeddings": self.primitive_embeddings}
        

        save_dict['attr_disentangler'] = self.attr_disentangler.state_dict()
        save_dict['obj_disentangler'] = self.obj_disentangler.state_dict()

        # save the soft embedding
        with torch.no_grad():
            if save_name == None:
                torch.save(save_dict, os.path.join(config.save_path, f"epoch_{epoch}.pt"))
            else:
                torch.save(save_dict, os.path.join(config.save_path,save_name))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):

        self.state_dict()['soft_embeddings'].copy_(state_dict['soft_embeddings'])
        self.image_text_encoder.load_state_dict(state_dict['image_text_encoder'])
        
        if 'primitive_embeddings' in state_dict:
            self.state_dict()['primitive_embeddings'].copy_(state_dict['primitive_embeddings'])

        self.attr_disentangler.load_state_dict(state_dict['attr_disentangler'])
        self.obj_disentangler.load_state_dict(state_dict['obj_disentangler'])
        
        return 
    