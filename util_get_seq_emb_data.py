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

def encode_core(core_smi, core_model, vocab):
    _, tensors, _ = tensorize([core_smi], vocab)
    tree_tensors, graph_tensors = make_cuda(tensors)
    root_vecs, tree_vecs, _, graph_vecs = core_model.encoder(tree_tensors, graph_tensors)
    root_vecs, root_kl = core_model.rsample(root_vecs, core_model.R_mean, core_model.R_var, perturb=False)

    return root_vecs

def encode_tail(tail_smi, tail_model, vocab):
    _, tensors, _ = tensorize([tail_smi], vocab)
    tree_tensors, graph_tensors = make_cuda(tensors)
    root_vecs, tree_vecs, _, graph_vecs = tail_model.encoder(tree_tensors, graph_tensors)
    root_vecs, root_kl = tail_model.rsample(root_vecs, tail_model.R_mean, tail_model.R_var, perturb=False)

    return root_vecs

def get_seq_emb_dataset():
    all_smi_list = []
    with open('qac-0315-clean.txt', "r") as f:
        for line in f:
            smi = line.strip()
            # Skip empty lines or whitespace-only lines
            if not smi:
                continue
            all_smi_list.append(smi)


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

    core_model = HierVAE(args_core).cuda()
    core_model.load_state_dict(torch.load("hgraph2graph/ckpt/cores/model.ckpt.1000")[0])

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

    tail_model = HierVAE(args_tail).cuda()
    tail_model.load_state_dict(torch.load("hgraph2graph/ckpt/tails/model.ckpt.800")[0])

    # Process, fragmentize, tensorize

    core_model.eval()
    tail_model.eval()
    with torch.no_grad():
        all_seqs = []
        for smi in tqdm(all_smi_list):

            latent_seq = []
            
            fragments = iterative_cut(smi)
            if len(fragments) == 0:
                print(smi)
                break

            # Process core
            core_smi = fragments[-1].strip("\r\n ").split()[0]
            latent_seq.append(encode_core(core_smi, core_model, core_vocab))
            
            # Process tails
            for tail_idx in range(len(fragments)-1):
                tail_smi = fragments[tail_idx]
                if Chem.MolFromSmiles(tail_smi) == None:
                    tail_smi = tail_smi.replace("n", "N")
                latent_seq.append(encode_tail(tail_smi, tail_model, tail_vocab))
            
            all_seqs.append(latent_seq)
    return all_seqs