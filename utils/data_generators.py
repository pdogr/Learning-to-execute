import random
import string
from utils.constants import EOL_TOK, EOS_TOK, PAD_TOK, SOS_TOK, TAB_TOK, UNK_TOK
import torchtext
from collections import Counter


class InverseGenerator():
    MAX_LEN = 20

    def __init__(self, batch_size, difficulty):
        self.batch_size = batch_size
        self.difficulty = difficulty
        self.specials = [UNK_TOK, PAD_TOK, EOS_TOK, PAD_TOK, SOS_TOK]
        self.vocab_syms = list(string.digits)+list(string.ascii_letters)
        self.src_vocab = torchtext.vocab.Vocab(
            Counter(self.specials+self.vocab_syms))
        self.tgt_vocab = torchtext.vocab.Vocab(
            Counter(self.specials+self.vocab_syms))

    def increase_difficulty(self):
        if self.difficulty < InverseGenerator.MAX_LEN:
            self.difficulty += 1

    def gen_list(self):
        len = random.randint(1, self.difficulty)
        return ''.join([random.choice(self.vocab_syms) for _ in range(len)])

    def next(self):
        inputs = [self.gen_list() for _ in range(self.batch_size)]
        inputs.sort(key=lambda x: len(x))
        target = [x[::-1] for x in inputs]
        return inputs, target


class AdditionGenerator():
    MAX_LEN = 7

    def __init__(self, batch_size, difficulty):
        self.difficulty = difficulty
        self.batch_size = batch_size
        self.specials = [SOS_TOK, PAD_TOK, UNK_TOK,
                         EOS_TOK, PAD_TOK, EOL_TOK, TAB_TOK]
        self.vocab_syms = list(string.digits+'+')
        self.src_vocab = torchtext.vocab.Vocab(
            Counter(self.specials+self.vocab_syms))
        self.tgt_vocab = torchtext.vocab.Vocab(
            Counter(self.specials+self.vocab_syms))

    def get_digit(self):
        return random.randint(1, 10**random.randint(1, self.difficulty))

    def increase_difficulty(self):
        if self.difficulty < AdditionGenerator.MAX_LEN:
            self.difficulty += 1

    def next(self):
        inputs = [
            (self.get_digit(), self.get_digit())
            for _ in range(self.batch_size)]
        inputs.sort(key=lambda x: len(str(x[0])+str(x[1])))
        target = [(x+y) for x, y in inputs]

        inputs = ["{0}+{1}".format(x, y) for x, y in inputs]
        target = [str(x) for x in target]

        return inputs, target
