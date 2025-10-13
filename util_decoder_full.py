from model_qacvae import GRUVAE
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_device('cuda')

# Decode back to molecule
import sys
sys.path.append("./hgraph2graph")

from hgraph2graph.hgraph import *
import argparse
import torch

from hgraph import HierVAE, common_atom_vocab
from hgraph.hgnn import make_cuda
from hgraph2graph.preprocess import tensorize
from hgraph import PairVocab
from util_split_core_tail import iterative_cut
from rdkit import Chem
from tqdm import tqdm

from util_reassemble_core_tail import attach_tails_to_core

class QACGenerator:
    def __init__(self):
        
        #####################
        # Load VAE Model    #
        #####################
        self.model = GRUVAE(input_size=8, hidden_size=100, latent_size=20, num_layers=1).cuda()
        state_dict = torch.load('/home/bo/HierQAC/pt_model.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        #####################
        # Load Core Encoder #
        #####################

        core_vocab_path = "hgraph2graph/vocab-cores.txt"
        with open(core_vocab_path) as f:
            core_vocab = [x.strip("\r\n ").split() for x in f]
        core_vocab = PairVocab(core_vocab, cuda=False)

        args_core = argparse.Namespace(
            seed=7,
            rnn_type='LSTM',
            hidden_size=100,   
            embed_size=100,    
            latent_size=8,
            depthT=15,
            depthG=15,
            diterT=1,
            diterG=3,
            dropout=0.0,
            vocab=core_vocab,
            atom_vocab=common_atom_vocab
        )

        self.core_model = HierVAE(args_core).cuda()
        self.core_model.load_state_dict(torch.load("hgraph2graph/ckpt/cores/model.ckpt.1000")[0])
        self.core_model.eval()

        #####################
        # Load Tail Encoder #
        #####################

        tail_vocab_path = "hgraph2graph/vocab-tails.txt"
        with open(tail_vocab_path) as f:
            tail_vocab = [x.strip("\r\n ").split() for x in f]
        tail_vocab = PairVocab(tail_vocab, cuda=False)

        args_tail = argparse.Namespace(
            seed=7,
            rnn_type='LSTM',
            hidden_size=100,   
            embed_size=100,    
            latent_size=8,
            depthT=15,
            depthG=15,
            diterT=1,
            diterG=3,
            dropout=0.0,
            vocab=tail_vocab,
            atom_vocab=common_atom_vocab
        )

        self.tail_model = HierVAE(args_tail).cuda()
        self.tail_model.load_state_dict(torch.load("hgraph2graph/ckpt/tails/model.ckpt.800")[0])
        self.tail_model.eval()

    def generate(self, z, seq_len):
        
        latent_dim = self.model.latent_size
        assert latent_dim == z.shape[1]
        n_samples = z.shape[0]

        samples = self.model.decode(z, seq_len)

        cores_z = samples[:, 0, :]     # Shape: (n_samples, latent_dim)
        tails_z = samples[:, 1:, :]    # Shape: (n_samples, seq_len - 1, latent_dim)

        core_mols = self.core_model.csample(cores_z, greedy=True)

        n_tails = seq_len - 1
        
        tails_z_reshaped = tails_z.reshape(n_samples * n_tails, self.tail_model.latent_size)

        tail_mols_flat = self.tail_model.csample(tails_z_reshaped, greedy=True)

        tail_mols_grouped = [
            tail_mols_flat[i*n_tails : (i+1)*n_tails]
            for i in range(n_samples)
        ]

        generated_smiles = []

        for i in range(n_samples):
            try:
                core_mol = core_mols[i]
                tails_for_this_core = tail_mols_grouped[i]  # 4 tails
                whole_mol = attach_tails_to_core(core_mol, tails_for_this_core)
                generated_smiles.append(whole_mol)
            except Exception as e:
                generated_smiles.append("")
                continue

        return generated_smiles
