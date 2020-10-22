from tempfile import tempdir
import torch
from .constants import EOL_TOK,SOS_TOK,TAB_TOK,EOS_TOK


def encode_input(sequence,vocab):
    encoded_input = []
    for seq in sequence:
        input_mapped = []
        seq_split=seq.split('\n')
        for i, seq_line in enumerate(seq_split):
            for seq_char in seq_line:
                if seq_char == '\t':
                    input_mapped.append(vocab.stoi[TAB_TOK])
                else:
                    input_mapped.append(vocab.stoi[seq_char])
        encoded_input.append(input_mapped)
    return encoded_input

def encode_target(sequence,vocab):
    encoded_input = []
    for seq in sequence:
        input_mapped = []
        input_mapped.append(vocab.stoi[SOS_TOK])
        for i, seq_line in enumerate(seq.split('\n')):
            for seq_char in seq_line:
                input_mapped.append(vocab.stoi[seq_char])
        input_mapped.append(vocab.stoi[EOS_TOK])
        encoded_input.append(input_mapped)
    return encoded_input


def toTensor(list,device,pad_id):
    tensor_list=[torch.tensor(x,dtype=torch.long,device=device) for x in list]
    return torch.nn.utils.rnn.pad_sequence(tuple(tensor_list),batch_first=True,padding_value=pad_id)

