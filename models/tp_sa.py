## Remove this later
import argparse
import os
from typing import Optional, Any, Union, Callable
from typing import List
import clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from models.common import  CrossResidualAttentionBlock,ResidualAttentionBlock
from utils.loss import multiple_label_ao_loss
import torch
from torch.nn import functional as F
from .encoder import TransformerEncoderLayer


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_tp_sa(train_dataset, config, device, prompt_template="a photo of x x"):
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
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))


    # get the token embedding of all attr object
    offset = len(attributes)
    offset2 = len(attributes) + len(classes)
    with torch.no_grad():
        primitive_embedding = torch.zeros(
            (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
        )

        for idx, rep in enumerate(orig_token_embedding[0:offset2]):
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
    ctx_vectors = ctx_vectors.repeat(2,1,1)

    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)




    primitive_embedding = nn.Parameter(primitive_embedding.to(device))
    soft_embedding = nn.Parameter(ctx_vectors).to(device) 

    # query_decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_width  ,nhead=8,batch_first=True)
    # query_based_decoder = nn.TransformerDecoder(decoder_layer=query_decoder_layer,num_layers=config.decoder_layer).to(device)

    encoder_width = soft_embedding.shape[-1]
    image_text_encoder_layer =  TransformerEncoderLayer(d_model = encoder_width,nhead=config.num_heads,batch_first = True)
    image_text_encoder = nn.TransformerEncoder(encoder_layer=image_text_encoder_layer, num_layers=config.num_encoder_layers).to(device)


    tpsa = TPSA(
        clip_model,
        config,
        offset,
        soft_embedding,
        primitive_embedding ,
        embedding_o,
        token_ids,
        device=device,
        enable_pos_emb=True,
        image_text_encoder = image_text_encoder,
        #attr_dropout=config.attr_dropout
    )

    tpsa = tpsa.to(device)

    optimizer = torch.optim.Adam(
        [
            {'params':tpsa.soft_embeddings},
            #{'params':tpsa.image_text_encoder.parameters()}

         ], 
         lr=config.lr, 
         weight_decay=config.weight_decay
    )


    if hasattr(config, "lr_2"):
        lr_2 = config.lr_2
    else:
        lr_2 = config.lr

    optimizer.add_param_group(
        {
            'params':tpsa.image_text_encoder.parameters(),
            'lr': lr_2,
            'weight_decay': config.weight_decay
        }
    )

    if config.train_vocab:
        optimizer.add_param_group(
            {
                'params': tpsa.primitive_embeddings,
                'lr': config.lr,
                'weight_decay': config.weight_decay
            }
        )
    
    if config.disent:
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



class TPSA(CLIPInterface):
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
        attr_dropout=0.3
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
        self.primitive_embeddings = primitive_embeddings

        self.offset = offset
        self.primitive_num = self.primitive_embeddings.shape[0]

        self.object_word_embedding = object_word_embedding

        self.image_text_encoder = image_text_encoder

        #self.query_token = nn.Parameter(torch.randn([1,self.frozen_attr_embeddings.shape[-1]])).to(self.device)
        #self.query_token.requires_grad = True
        #self.cls_layer = nn.Parameter(torch.randn())
        self.multi_loss_func_name = config.multi_label_loss


        self.train_vocab = config.train_vocab

        self.attr_dropout = nn.Dropout(attr_dropout)
        print(self.multi_loss_func_name)

        if hasattr(config, "disent") and config.disent:
            self.attr_disentangler = Disentangler(primitive_embeddings.shape[-1])
            self.obj_disentangler = Disentangler(primitive_embeddings.shape[-1])
            self.disent = True
        else:
            self.disent = False

        if hasattr(config,'refined_text') :
            print('model.refined_text',config.refined_text)

        if hasattr(config,'merge_patch') :
            print('model.merge_patch',config.refined_text)

    def construct_token_tensors(self):

        # repeat 'a photo of x x' for s+o times
        class_token_ids = self.token_ids.repeat(self.primitive_num, 1)

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

        # token embedding for objs: [sot][x1][x2][x3][o][eot]
        token_tensor[self.offset:self.primitive_num , eos_idx - 2, :] = primitive_embeddings[self.offset:].type(self.clip_model.dtype)
        token_tensor[self.offset:self.primitive_num , eos_idx - 1, :] = eos_token_embedding
        token_tensor[self.offset:self.primitive_num , eos_idx , :] = none_token_embedding

        # add learnable prompt embedding: s/o/c
        len_soft_embedding = len(self.soft_embeddings[0])
        token_tensor[
            0:self.offset, 1 : len_soft_embedding + 1, :
        ] = self.soft_embeddings[0].type(self.clip_model.dtype).squeeze(0)

        token_tensor[
            self.offset:self.primitive_num, 1 : len_soft_embedding + 1, :
        ] = self.soft_embeddings[1].type(self.clip_model.dtype).squeeze(0)

        return token_tensor


    def encode_image(self, imgs,return_patch=True):
        return self.clip_model.encode_image(imgs,return_patch=return_patch)
    
    
    def logit_infer_multi_attr(self, predict, pairs):
        attr_logits, obj_logits = predict

        # attr_logits = attr_logits.softmax(dim=-1)
        # obj_logits = obj_logits.softmax(dim=-1)


        comp_logits = torch.zeros([attr_logits.shape[0],len(pairs)]).to(attr_logits.device)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_logits = 0 if self.config.attr_inference_weight == 0 else attr_logits[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_logits = 0 if self.config.obj_inference_weight == 0 else obj_logits[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = weighted_attr_logits + weighted_obj_logits

        return comp_logits

    def forward(self, batch_img, idx=None,text_features=None):
        
        if isinstance(batch_img,list):
            batch_img = batch_img[0].to(self.device)
        else:
            batch_img = batch_img.to(self.device)

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

        num_image_token = patch_features.shape[1]

        num_text_token =  self.primitive_num

        # convert to float32 for the following module
        text_features = text_features.float()
        batch_img, patch_features = batch_img.float(), patch_features.float()

        num_text_token =  self.primitive_num
        num_attr_token = self.offset
        num_obj_token = num_text_token - num_attr_token

        # forward with attribute/object/image text encoder
        if not self.disent:


            encoder_input = torch.cat( 
                    [batch_img.unsqueeze(1),
                    patch_features, 
                    text_features.repeat(batch,1,1)] 
                ,dim=1)
            

            # The input tensor sequence is [CLS, Patch Features, Attribute Tokens, Object Tokens]
            # 1. Define the start and end indices for attribute and object tokens
            # Sequence: [CLS (1), Patches (num_image_token), Attrs (num_attr_token), Objs (num_obj_token)]
            total_seq_len = 1 + num_image_token + num_text_token
            attr_start_idx = 1 + num_image_token
            attr_end_idx = attr_start_idx + num_attr_token
            obj_start_idx = attr_end_idx
            
            # 2. Create the attention mask. In PyTorch's TransformerEncoder, a value of 0 allows attention,
            #    while -inf prevents it. We will create a boolean mask first (True=allow, False=block)
            #    and then convert it.
            attention_mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool, device=self.device)
            

            if hasattr(self.config, "mask") and self.config.mask == "attr_obj":
                # 3. Block attention between attribute and object tokens
                # Block attributes from attending to objects
                # print('use mask',self.config.mask)
                attention_mask[attr_start_idx:attr_end_idx, obj_start_idx:] = True
                # Block objects from attending to attributes
                attention_mask[obj_start_idx:, attr_start_idx:attr_end_idx] = True
            elif hasattr(self.config, "mask") and self.config.mask == "primitive":
                attention_mask[attr_start_idx:, attr_start_idx:] = True
            elif hasattr(self.config, "mask") and self.config.mask == "image_text":
                # Block image from attending to text
                attention_mask[0:num_image_token, attr_start_idx:] = True
                # Block text from attending to image
                attention_mask[attr_start_idx:, 0:num_image_token] = True
            elif hasattr(self.config, "mask") and self.config.mask == "cls_patch":
                attention_mask[0, 1:num_image_token] = True
                attention_mask[1:num_image_token , 0] = True
            elif hasattr(self.config, "mask") and self.config.mask == "primitive_respective":
                attention_mask[attr_start_idx:attr_end_idx, attr_start_idx:attr_end_idx] = True
                attention_mask[obj_start_idx: , obj_start_idx:] = True
            
            img_text_features = self.image_text_encoder(encoder_input, mask=attention_mask)
            
            cls_fs,patch_fs,encoded_text_fs = torch.split(img_text_features,[1,num_image_token,num_text_token],dim=1)
        


            cls_fs = cls_fs.squeeze(1)
            if hasattr(self.config, 'merge_patch') and self.config.merge_patch == True:
                fuser_patch = patch_fs.mean(dim=1,keepdim=False)  # patch_fs: B*N*C

                cls_fs = cls_fs + fuser_patch

            
            cls_fs = cls_fs / cls_fs.norm(dim=-1,keepdim=True)


            if hasattr(self.config,'refined_text') and self.config.refined_text == True:
                encoded_text_fs = encoded_text_fs / encoded_text_fs.norm(dim=-1,keepdim=True)

                logits = torch.einsum('BC,BNC->BN', cls_fs, encoded_text_fs) * self.clip_model.logit_scale.exp()

            else:
                text_features = text_features / text_features.norm(dim=-1,keepdim=True)
                logits = (
                    self.clip_model.logit_scale.exp()
                    * cls_fs
                    @ text_features.t()
                )

            attr_logits, obj_logits  = logits[:,0:self.offset] ,logits[:,self.offset:self.primitive_num]    



        return [attr_logits,obj_logits]


    def forward_attn_map(self, batch_img, idx=None,text_features=None, attention_mask=None): # Note: Added attention_mask to signature
            
            if isinstance(batch_img,list):
                batch_img = batch_img[0].to(self.device)
            else:
                batch_img = batch_img.to(self.device)

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

            num_image_token = patch_features.shape[1]
            num_text_token =  self.primitive_num

            # convert to float32 for the following module
            text_features = text_features.float()
            batch_img, patch_features = batch_img.float(), patch_features.float()

            num_text_token =  self.primitive_num
            num_attr_token = self.offset
            num_obj_token = num_text_token - num_attr_token

            # forward with attribute/object/image text encoder
            if not self.disent:
                encoder_input = torch.cat( 
                        [batch_img.unsqueeze(1),
                        patch_features, 
                        text_features.repeat(batch,1,1)] 
                    ,dim=1)
                
                # The input tensor sequence is [CLS, Patch Features, Attribute Tokens, Object Tokens]
                total_seq_len = 1 + num_image_token + num_text_token
                attr_start_idx = 1 + num_image_token
                attr_end_idx = attr_start_idx + num_attr_token
                obj_start_idx = attr_end_idx

                # --- MODIFICATION START ---
                # Manually iterate through the encoder layers to extract the attention map
                # from the last layer.

                x = encoder_input
                attention_map = None  # Initialize attention_map
                
                # Get the number of layers from the encoder
                num_layers = len(self.image_text_encoder.layers)
                
                # 1. Iterate through all layers EXCEPT the last one
                for i in range(num_layers - 1):
                    # Pass the attention_mask to each layer as 'src_mask'
                    x = self.image_text_encoder.layers[i](x, src_mask=attention_mask)
                    
                # 2. Call the LAST layer with `need_weights=True`
                # This makes the layer return a tuple: (output, attention_weights)
                x, attention_map = self.image_text_encoder.layers[-1](x, src_mask=attention_mask, need_weights=True)
                
                # 3. Manually apply the final normalization layer from the encoder, if it exists
                # This step is crucial because the loop bypasses the encoder's final norm layer.
                if self.image_text_encoder.norm:
                    img_text_features = self.image_text_encoder.norm(x)
                else:
                    img_text_features = x
                # --- MODIFICATION END ---
                
                cls_fs,patch_fs,encoded_text_fs = torch.split(img_text_features,[1,num_image_token,num_text_token],dim=1)
                
                cls_fs = cls_fs.squeeze(1)
                cls_fs = cls_fs / cls_fs.norm(dim=-1,keepdim=True)

                if hasattr(self.config,'refined_text') and self.config.refined_text == True:
                    encoded_text_fs = encoded_text_fs / encoded_text_fs.norm(dim=-1,keepdim=True)
                    logits = torch.einsum('BC,BNC->BN', cls_fs, encoded_text_fs) * self.clip_model.logit_scale.exp()
                else:
                    text_features = text_features / text_features.norm(dim=-1,keepdim=True)
                    logits = (
                        self.clip_model.logit_scale.exp()
                        * cls_fs
                        @ text_features.t()
                    )

                attr_logits, obj_logits  = logits[:,0:self.offset] ,logits[:,self.offset:self.primitive_num]  

            # --- MODIFICATION TO RETURN VALUE ---
            # The function now returns the logits and the captured attention map.
            return [attr_logits,obj_logits], attention_map


    def multi_attr_loss_calu(self, predict, target,pair_positive_weight=None,attr_positive_weight=None):
        
        loss = multiple_label_ao_loss(predict=predict,
                                    target=target,
                                    pair_positive_weight=pair_positive_weight,
                                    attr_positive_weight=attr_positive_weight,
                                    multi_loss_func_name=self.config.multi_label_loss)

        return loss
    
    def single_attr_loss_calu(self, predict, target):
        
        loss_fn = nn.CrossEntropyLoss()
        single_attr_loss_fn = nn.CrossEntropyLoss()

        _, batch_attr, batch_obj, batch_target = target
        attr_logits, obj_logits = predict[0:3]

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        

        loss_attr = single_attr_loss_fn(attr_logits, batch_attr)

        loss_obj = loss_fn(obj_logits, batch_obj)

        loss = loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight
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
        return self.forward(batch,text_features=text_feats)


    def save_params(self,config,epoch,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        save_dict = {"soft_embeddings":self.soft_embeddings,
                     "image_text_encoder":self.image_text_encoder.state_dict(),
                     "primitive_embeddings": self.primitive_embeddings}
        
        if self.disent:
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
        
        if 'primitive_embeddings' in state_dict and self.train_vocab:
            self.state_dict()['primitive_embeddings'].copy_(state_dict['primitive_embeddings'])
        
        if self.disent:
            self.attr_disentangler.load_state_dict(state_dict['attr_disentangler'])
            self.obj_disentangler.load_state_dict(state_dict['obj_disentangler'])
        
        return 
    