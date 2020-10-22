import argparse
import os
import csv
import random
from string import ascii_letters, digits, punctuation

import utils.constants as CONSTANT
from utils.sequence import *
from utils.data_generators import InverseGenerator
from dataloader import Dataloader

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    d=Dataloader(device=device,batch_size=12,difficulty=1)
    for (input,input_lens,target) in d:
        print(input,input_lens)
        break


