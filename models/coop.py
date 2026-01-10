## Remove this later
import argparse
import os

import clip
import torch
import torch.nn as nn
from utils.loss import multiple_label_c_loss
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def coop(train_dataset, config, device, prompt_template="a photo of x x"):
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]


    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )   
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device)) # num x context_length x dim 

    # frozen_embedding is the clip world embedding of attribute/object  
    with torch.no_grad():
        frozen_embedding = torch.zeros(
            (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
        ).to(device)
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()       # the index of <eot>  
            frozen_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init,
                           context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)

    ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

    # the token_id for prompt template
    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)
    



    # shape: 3*768
    soft_embedding = nn.Parameter(ctx_vectors).to(device)

    optimizer = torch.optim.Adam(
        [soft_embedding], lr=config.lr, weight_decay=config.weight_decay
    )
    offset = len(attributes)

    coop = COOP(
        clip_model,
        config,
        offset,
        soft_embedding,
        frozen_embedding,
        token_ids,
        device=device,
        enable_pos_emb=True,
    )

    return coop, optimizer


class COOP(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: torch.nn.Parameter,
        frozen_embeddings: torch.nn.Parameter,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.frozen_embeddings = frozen_embeddings
        self.offset = offset

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)


        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = self.frozen_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)

        token_tensor[:, eos_idx - 1, :] = self.frozen_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)


        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_embeddings) + 1, :
        ] = self.soft_embeddings.type(self.clip_model.dtype)

        return token_tensor


    def forward(self, batch, idx):

        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip_model.dtype))
        token_tensors = self.construct_token_tensors(idx)

        text_features = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * batch_img
            @ text_features.t()
        )

        return logits
    

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)

        batch_size = 1024
        text_features = torch.Tensor().to(self.device).type(self.clip_model.dtype)
        for i in tqdm(range(0, len(token_tensors), batch_size)):
            batch_token_tensors = token_tensors[i:i+batch_size]
            batch_text_features = self.text_encoder(
            self.token_ids,
            batch_token_tensors,
            enable_pos_emb=self.enable_pos_emb,
            )
            
            text_features = torch.cat([text_features,batch_text_features],dim=0)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    
    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip_model.dtype))

        text_features = text_feats
        batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * batch_img
            @ text_features.t()
        )

        return logits

    def multi_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        

        loss = multiple_label_c_loss(predict=predict,
                                     target=target,
                                      multi_loss_func_name=self.config.multi_label_loss)
        return loss
    
    def single_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
        loss_fn = nn.CrossEntropyLoss()

        _, batch_attr, batch_obj, batch_pair = target
        comp_logits = predict

        batch_pair = batch_pair.cuda()
        
        loss_comp = loss_fn(comp_logits, batch_pair)

        loss = loss_comp
        return loss

    def logit_infer_multi_attr(self,predict,pairs,topk=3):
        comp_logits = predict

        return comp_logits
    

    
    def save_params(self,config,epoch,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        # save the soft embedding
        with torch.no_grad():
            if save_name == None:
                torch.save({"soft_embeddings":self.soft_embeddings}, os.path.join(config.save_path, f"epoch_{epoch}.pt"))
            else:
                torch.save({"soft_embeddings":self.soft_embeddings}, os.path.join(config.save_path, save_name))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):

        self.state_dict()['soft_embeddings'].copy_(state_dict['soft_embeddings'])
        return 