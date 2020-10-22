import torch
import torchtext
from torchtext.data.iterator import BucketIterator
from base import BaseDataLoader
from utils.constants import PAD_TOK, EOS_TOK, SOS_TOK, EOL_TOK, TAB_TOK
from utils.sequence import encode_input, encode_target, toTensor
from utils.data_generators import *
import random

class Dataloader():
    def __init__(
            self, batch_size, device, difficulty,
            type='inverse',
            src_preprocessing=None, tgt_preprocessing=None,
            train=True):
        self.device = device
        self.batch_size=batch_size
        self.type=type
        self._gen = None
        if type=='inverse':
            self._gen=InverseGenerator(self.batch_size,difficulty)
        elif type=='add':
            self._gen=AdditionGenerator(self.batch_size,difficulty)

        self._src_vocab = self.gen.src_vocab
        self._tgt_vocab = self.gen.tgt_vocab
        self.train = train

    def increase_difficulty(self):
        self.gen.increase_difficulty()

    @property
    def src_vocab(self):
        return self._src_vocab

    @property
    def tgt_vocab(self):
        return self._tgt_vocab

    @property
    def gen(self):
        return self._gen

    def __next__(self):
        inputs, target = self.gen.next()
        inputs_encoded = encode_input(inputs,self.src_vocab)
        input_lens = [len(x) for x in inputs_encoded]
        target_encoded = encode_target(target,self.tgt_vocab)

        inputs_encoded = toTensor(inputs_encoded, self.device,self.src_vocab.stoi[PAD_TOK])
        input_lens = torch.LongTensor(input_lens).to(self.device)
        target_encoded = toTensor(target_encoded, self.device,self.tgt_vocab.stoi[PAD_TOK])
        return (inputs_encoded, input_lens, target_encoded)

    def __iter__(self):
        return self
