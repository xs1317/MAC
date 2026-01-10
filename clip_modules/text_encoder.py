import clip
import torch


class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)

        text_batch = 256
        
        # if token_ids.shape[0] > text_batch:
        #     text_features = torch.zeros([token_ids.shape[0],])
        #     for tk in torch.split(token_ids,text_batch,dim = 0):
        
        
        text_features = self.forward(token_ids, None, enable_pos_emb)



        return text_features

    def forward(self, token_ids, token_tensors=None, enable_pos_emb=False,return_token=False):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        
        if token_tensors is not None:
            input_dim = token_tensors.ndim
            if token_tensors.ndim == 4:  # [B, A, L, D]
                B, A, L, D = token_tensors.shape
                token_tensors = token_tensors.view(B * A, L, D)
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids) 

        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )

        
        # text_batch = 512
        # text_features = torch.Tensor().to(x.device).type(self.dtype)
        # if x.shape[0] > text_batch :
        #     for i, batch_x in enumerate(torch.split(x,text_batch,dim=0)):
        #         batch_x = batch_x.permute(1, 0, 2)      # NxLxD --> LxNxD
        #         batch_x = self.transformer(batch_x)
        #         batch_x = batch_x.permute(1, 0, 2)
        #         batch_x = self.ln_final(batch_x)
        #         text_features = torch.cat([text_features, batch_x], dim=0)
        #     x = text_features
        # else:



        
        x = x.permute(1, 0, 2)      # NxLxD --> LxNxD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)


        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]  # POS of <EOS>
            @ self.text_projection
        )    
        
        if token_tensors is not None and input_dim == 4:
            tf = tf.view(B, A, -1)  # [B, A, D]


        # if return_token == False:
        return tf
        # else:
        #     return tf, x
