from torch.utils.data import Dataset, DataLoader
import pickle, torch, json, os, sys, re
from sys import stdout
from os.path import join, exists, abspath, dirname
import numpy as np  
import random
from collections import Counter

curpath = dirname(abspath(__file__))
upperpath = dirname(curpath)
sys.path.append(upperpath)

def pad_seqs(seqs, tensorType):
      """ 
      This function pads seqeunces of variable lengths to a fixed length 

      Args:
        seqs: list of list of numbers 

        tensor_type: 
          the tensor type of the sequence 
 
      """

      batchSize = len(seqs)

      seqLenths = torch.LongTensor(list(map(len, seqs)))
      maxSeqLen = seqLenths.max()

      seqTensor = torch.zeros(batchSize, maxSeqLen, dtype = tensorType)

      mask = torch.zeros(batchSize, maxSeqLen, dtype = torch.long)

      for i, (seq, seqLen) in enumerate(zip(seqs, seqLenths)):
        seqTensor[i,:seqLen] = torch.tensor(seq, dtype = tensorType)
        mask[i,:seqLen] = torch.LongTensor([1]*int(seqLen))
      return seqTensor, mask


def collate(batch):
    
    batch_size = len(batch)

    item = {}

    sent_inputids, sent_attention_masks = pad_seqs([ins.inputIds for ins in batch], torch.long)
    item["sent_inputids"] = sent_inputids
    item["sent_attention_masks"] = sent_attention_masks
    if hasattr(batch[0], 'bertStartIdx'):
        item["bert_starts"] = torch.LongTensor([ins.bertStartIdx for ins in batch])
        item["bert_ends"] = torch.LongTensor([ins.bertEndIdx for ins in batch])
    item["labels"] = torch.LongTensor([ins.intRelationLabel for ins in batch]) if batch[0].intRelationLabel is not None else torch.LongTensor([-1] * batch_size)
    item["instances"] = batch

    return item 

class MyDataset(Dataset):
    def __init__(self,instances):

        super(Dataset, self).__init__()

        # first shuffle
        self.instances = instances
        random.Random(100).shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]
