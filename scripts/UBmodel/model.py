from transformers import * 
import torch 
import sys
from os.path import join, dirname, abspath

curpath = dirname(abspath(__file__))
upperpath = dirname(curpath)
sys.path.append(upperpath)

from transformers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnaryModel(torch.nn.Module):
  def __init__(self, tokenizer, args): 
    super(UnaryModel, self).__init__()

    self.modelType = args.modelType
    if self.modelType in ["bert-large-uncased", "bert-large-cased"]:
      self.encoder_dimension = 1024
    else:
      self.encoder_dimension = 768

    self.bert_encoder = AutoModel.from_pretrained(self.modelType)
    self.tokenizer = tokenizer
    self.bert_encoder.resize_token_embeddings(len(tokenizer))

    self.relu = torch.nn.LeakyReLU()
    self.dropout = torch.nn.Dropout(args.dropout)

    self.layernorm = torch.nn.LayerNorm(self.encoder_dimension)
    self.linear = torch.nn.Linear(self.encoder_dimension, 2)


  def forward(self,item, mask_local = False, mask_k = 10):
      sent_inputids = item["sent_inputids"].to(device)
      sent_attention_masks = item["sent_attention_masks"].to(device)
      starts1 = item["bert_starts"].to(device)
      ends1 = item["bert_ends"].to(device)
      out = self.bert_encoder(sent_inputids, attention_mask = sent_attention_masks)
      last_hidden_states = out[0]
      batch_size, seq_len, dimension = last_hidden_states.size() # get (batch, 2*dimension), [start_embedding, end_embedding]
      start_indices1 = starts1.view(batch_size, -1).repeat(1, dimension).unsqueeze(1)# (batch, 1, dimension)
      start_embeddings1 = torch.gather(last_hidden_states, 1, start_indices1).view(batch_size, -1) # shape (batch, dimension)
      end_indices1 = ends1.view(batch_size, -1).repeat(1, dimension).unsqueeze(1) # (batch, 1, dimension) 
      end_embeddings1 = torch.gather(last_hidden_states, 1, end_indices1).view(batch_size, -1) # shape (batch, dimension)
      norm_rep = start_embeddings1
      norm_rep = self.relu(norm_rep)
      norm_rep = self.dropout(norm_rep)
      logits =  self.linear(norm_rep)

      return logits
