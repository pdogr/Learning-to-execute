from string import ascii_lowercase, digits
from torchtext.vocab import Vocab
from collections import Counter
SOS_TOK = '<sos>'
PAD_TOK = '<pad>'
EOS_TOK = '<eos>'
EOL_TOK = '<eol>'
TAB_TOK = '<tab>'
UNK_TOK = '<unk>'
LETTERS = ascii_lowercase+digits+'*.-+/='

SYMBOLS = [PAD_TOK, TAB_TOK, EOL_TOK] + list(LETTERS)
TGT_SYMBOLS = [PAD_TOK, SOS_TOK, EOS_TOK] + list(digits)+['-']
