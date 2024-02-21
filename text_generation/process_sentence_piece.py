import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter

import regex as re
import os
import time
from tqdm import tqdm
import copy
import math

from .transformer_model import TransformerModel
from .model_utils import preProcessText, getTokenizer,try_gpu 
from .model_config import getConfig



def get_model(model_config, ntokens):
    emsize = model_config["emsize"]
    d_hid = model_config["d_hid"]
    nlayers = model_config["nlayers"]
    nhead = model_config["nhead"]
    dropout = model_config["dropout"]
    model = TransformerModel(ntokens, emsize,nhead, d_hid, nlayers, dropout)
    return model

model_config, app_config = getConfig()


bptt=model_config["bptt"]

device = "cuda" if torch.cuda.is_available() else "cpu"

softmax = nn.Softmax(dim=2)

tokenizer, vocab = getTokenizer(tokenizer_type = "sentence_piece")
ntokens = len(vocab)
model = get_model(model_config, ntokens).to(device)

def loadModel(best_model_path):
    if os.path.exists(best_model_path):
        print(f"Preloading model {best_model_path}")
        if torch.cuda.is_available():
            state = torch.load(best_model_path)
        else:
            state = torch.load(best_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
        return model
    else:
        raise Exception("Model Not Found")
    
best_model_path = 'models/model_sentence_piece.pt'
loaded_model = loadModel(best_model_path)

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(tokenizer.encode(item).ids, dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, batch_size: int) -> Tensor:
    """Divides the data into batch_size separate sequences, removing extra elements
    that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N]
        batch_size: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(device)

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def split_list(l):
    splitted_list = []
    z = 0
    for i,idx in enumerate(l):
        if idx == 220:
            splitted_list.append(l[z:i])
            z = i+1
    if z <= len(l)-1:
        splitted_list.append(l[z:])
    return splitted_list

def splits_to_token(splited_list):
    strings = [tokenizer.decode(l) for l in splited_list]
    
    return strings

def nonnaive_generator(model: nn.Module, gen_data: Tensor, no_words = 5,k=50):
    model.eval()

    src_mask = generate_square_subsequent_mask(bptt).to(device)
    pred_text = []
    for i in range(no_words):
        batch_size = gen_data.size(0)
        if batch_size != bptt:
            src_mask_ = src_mask[:batch_size, :batch_size]
        output_softmax = model(gen_data, src_mask_)
        output_softmax_permuted = output_softmax.permute(1, 0, 2)
        indices = torch.topk(output_softmax_permuted,k ,dim=2).indices.squeeze(0)
        
        values = torch.topk(softmax(output_softmax_permuted),k ,dim=2).values
        values = values/torch.sum(values,dim = 2,keepdims = True)
        
        ind_sampled = torch.distributions.Categorical(values.squeeze(0)).sample()
        next_index = indices[-1][ind_sampled[-1]]

        pred_text.append(next_index.item())
        if(batch_size < 128):
            gen_data = torch.cat((gen_data[:,:],next_index.unsqueeze(0).unsqueeze(0)),0)
            batch_size= gen_data.size(0)
        else:
            gen_data = torch.cat((gen_data[1:,:],next_index.unsqueeze(0).unsqueeze(0)),0)
            batch_size= gen_data.size(0)
            
    return pred_text

def predTextSentencePiece(text : str, num_words : int):
    text = [text]
    num_words = int(num_words)
    sample_data = data_process(text)
    sample_data = batchify(sample_data, 1)
    pred_text = nonnaive_generator(loaded_model,  sample_data[:,-1].unsqueeze(1), no_words=num_words, k=50)
    m = splits_to_token(split_list(pred_text))
    return ' '.join(text)+' ' +' '.join(m)
