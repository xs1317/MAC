import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from utils.loss import multiple_label_c_loss
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, activation='relu'):
        """
        :param input_dim: 输入特征维度
        :param hidden_dims: list，如 [512, 256]
        :param output_dim: 输出特征维度
        :param dropout: dropout 概率
        :param activation: 激活函数类型，支持 relu / leaky_relu / gelu
        """
        super(MLP, self).__init__()

        layers = []
        dims = [input_dim] + hidden_dims
        act_fn = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU()
        }[activation]

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))  # Output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

        if hasattr(self.config, 'logit_scale'):
            self.clip_model.logit_scale.data.fill_(self.config.logit_scale)
        
        print(self.clip_model.logit_scale)

        if hasattr(self.config, 'image_net') and config.image_net == True:
            self.image_net = True
            clip_model_dim = soft_embeddings.shape[-1]
            self.image_net = MLP(input_dim = clip_model_dim,hidden_dims=[clip_model_dim]*1,output_dim=clip_model_dim)
        else:
            self.image_net = False

    def construct_token_tensors(self, pair_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        return  token_tensor
    
    
    def forward(self, batch, idx):
        token_tensors = self.construct_token_tensors(idx)




        # 合并所有 batch 的输出结果
        text_features = self.text_encoder(
                self.token_ids,
                token_tensors,
                enable_pos_emb=self.enable_pos_emb,
            )


        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip_model.dtype))

        if self.image_net:
            batch_img = batch_img.float()
            text_features = text_features.float()
            batch_img = batch_img + self.image_net(batch_img)

        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # 计算 logits
        logits = (
            self.clip_model.logit_scale.exp() *
            batch_img @ text_features.t()
        )

        return logits
    
    def multi_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
        
        loss = multiple_label_c_loss(predict=predict,
                                     target=target,
                                     pair_positive_weight=pair_positive_weight,
                                     attr_positive_weight=attr_positive_weight,
                                      multi_loss_func_name=self.config.multi_label_loss)
        return loss
    
    def single_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
        _, _, _, batch_target = target
        comp_logits = predict

        batch_target = batch_target.cuda()


        loss_fn = nn.CrossEntropyLoss()
 

        loss_comp = loss_fn(comp_logits, batch_target)

        loss = loss_comp 
        
        return loss


    def logit_infer_multi_attr(self,predict,pairs,topk=3):
        comp_logits = predict

        return comp_logits


    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)

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

        text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        return text_features

    
    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip_model.dtype))
        text_features = text_feats
        batch_img = batch_img.float()
        text_features = text_features.float()
        if self.image_net:
            batch_img = batch_img +  self.image_net(batch_img)

        
    
        batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()
            * batch_img
            @ text_features.t()
        )

        return logits
    
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):

        #print(state_dict.keys())
        self.state_dict()['soft_embeddings'].copy_(state_dict['soft_embeddings'])

        if self.image_net:
            self.attr_disentangler.load_state_dict(state_dict['image_net'])

        return 
    
    def save_params(self,config,epoch,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        save_dict = {"soft_embeddings":self.soft_embeddings}

        if self.image_net:
            save_dict[ 'image_net'] = self.image_net.state_dict()
        # save the soft embedding
        with torch.no_grad():
            if save_name == None:
                torch.save(save_dict, os.path.join(config.save_path, f"epoch_{epoch}.pt"))
            else:
                torch.save(save_dict, os.path.join(config.save_path, save_name))


def csp_init(
    train_dataset,
    config,
    device,
    prompt_template="a photo of X X",
):

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

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),device=device
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    soft_embedding = soft_embedding.to(device)
    soft_embedding = nn.Parameter(soft_embedding)
    


    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    )



def get_csp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)



    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )



    optimizer = torch.optim.Adam(
        [
            {'params':soft_embedding}
         ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if hasattr(config,'image_net') and  config.image_net:
        optimizer.add_param_group(
            {
                'params': interface.image_net,
                'lr': config.lr,
                'weight_decay': config.weight_decay
            }
        )

    interface = interface.to(device)

    return interface, optimizer


def get_mix_csp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    with torch.no_grad():
        subset_soft_embeddings = soft_embedding[train_dataset.indices, :]

    subset_soft_embeddings.requires_grad = True

    optimizer = torch.optim.Adam(
        [subset_soft_embeddings],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # reduce the offset to selected offset
    offset = len(train_dataset.attr_indices)
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        subset_soft_embeddings,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface, optimizer
