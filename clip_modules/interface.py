import argparse

import torch
from clip.model import CLIP

from .text_encoder import CustomTextEncoder
import os

class CLIPInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        token_ids: torch.tensor,
        soft_embeddings = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            token_ids (torch.tensor): the input token ids to the text
                encoder
            soft_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.enable_pos_emb = enable_pos_emb

        self.text_encoder = CustomTextEncoder(clip_model, self.dtype)
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        self.token_ids = token_ids
        self.soft_embeddings = soft_embeddings

    def encode_image(self, imgs,return_patch=False):
        return self.clip_model.encode_image(imgs,return_patch=return_patch)

    def encode_text(self, text, enable_pos_emb=True):
        return self.text_encoder.encode_text(
            text, enable_pos_emb=enable_pos_emb
        )

    def tokenize(self, text):
        return self.text_encoder.tokenize(text)

    def set_soft_embeddings(self, se):
        if not isinstance(se,list) and se.shape == self.soft_embeddings.shape:
            self.state_dict()['soft_embeddings'].copy_(se)
        elif isinstance(se,list) and se[0].shape == self.soft_embeddings[0].shape :
            self.soft_embeddings = se
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!")

    def construct_token_tensors(self, idx):
        """The function is used to generate token tokens. These
        token tensors can be None or custom. For custom token_tensors
        the class needs to be inherited and the function should be
        replaced.

        Raises:
            NotImplementedError: raises error if the model contains
            soft embeddings but does not make custom modifications.

        Returns:
            torch.Tensor: returns torch.Tensor or None
        """
        if self.soft_embeddings is None:
            return None
        else:
            # Implement a custom version
            raise NotImplementedError

    # def forward(self, batch_img, idx):
    #     batch_img = batch_img.to(self.device)

    #     token_tensors = self.construct_token_tensors(idx)

    #     text_features = self.text_encoder(
    #         self.token_ids,
    #         token_tensors,
    #         enable_pos_emb=self.enable_pos_emb,
    #     )


    #     # #_text_features = text_features[idx, :]
    #     # _text_features = text_features

    #     # idx_text_features = _text_features / _text_features.norm(
    #     #     dim=-1, keepdim=True
    #     # )
    #     # normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
    #     # logits = (
    #     #     self.clip_model.logit_scale.exp()
    #     #     * normalized_img
    #     #     @ idx_text_features.t()
    #     # )

    #     text_features = text_features / text_features.norm(
    #         dim=-1, keepdim=True
    #     )
    #     batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
    #     logits = (
    #         self.clip_model.logit_scale.exp()
    #         * batch_img
    #         @ text_features.t()
    #     )


    #     return logits
        
    def forward(self, batch, token_tensors):

        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip_model.dtype))
        #token_tensors = self.construct_token_tensors(idx)

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
    

    def save_params(self,config,epoch,save_dict,save_name=None):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        # save the soft embedding
        with torch.no_grad():
            if save_name == None:
                torch.save(save_dict, os.path.join(config.save_path, f"epoch_{epoch}.pt"))
            else:
                torch.save(save_dict, os.path.join(config.save_path, save_name))
