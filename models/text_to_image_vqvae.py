import torch
import torch.nn as nn
import numpy as np
from models.vqvae import VQVAE
from models.quantizer import VectorQuantizer

class Grounded_VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 codebook_dict, decoder_dict):
        super(grounded_vqvae, self).__init__()
        # Initialize the codebook and decoder
        self.codebook = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder  = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def _load_from_pretrained(self, codebook_dict, decoder_dict):
        # Load in the pre-trained decoder and codebook
        self.decoder.load_state_dict(state_dict)

class Fusion_VQVAE(nn.Module):
