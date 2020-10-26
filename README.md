# Learning-to-execute

This repo contains the attention based implementation of ideas devised in the paper [Learning to Execute](https://arxiv.org/abs/1410.4615) with [PyTorch](https://github.com/pytorch/pytorch) and [TorchText](https://github.com/pytorch/text) using Python 3.8.
.

## Example outputs
### Reversing task

Output             |  Attn Wts
:-------------------------:|:-------------------------:
![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/reverse_example/r1.png)  |  ![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/reverse_example/r1_attn_wts.png)
![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/reverse_example/r2.png)  |  ![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/reverse_example/r2_attn_wts.png)

### Addition task

Output             |  Attn Wts
:-------------------------:|:-------------------------:
![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/add_example/a1.png)  |  ![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/add_example/a1_attn_wts.png)
![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/add_example/a2.png)  |  ![](https://github.com/plaxi0s/Learning-to-execute/blob/master/examples/add_example/a2_attn_wts.png)


## Getting Started

#### Install torch and torchtext
To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torch torchtext
```

#### Train
The parameters for training can be configured via config.json. By default trainined models, logs, tensorboard logs are stored in saved/ directory.

```bash
python3 train.py -h
usage: train.py [-h] [-c CONFIG] [-r RESUME] -d DIFFICULTY -t {inverse,add} [-de DEVICE] [--lr LR]
                [--bs BS]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DIFFICULTY, --difficulty DIFFICULTY
                        Change difficulty
  -t {inverse,add}, --type {inverse,add}
                        Change type (choice)
  -de DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --lr LR, --learning-rate LR
                        Change optimzer learning_rate
  --bs BS, --batch-size BS
                        Change batch_size of dataloader
```

#### Test
After training you can check the model for predictions via entering a sequence
``` bash
usage: test.py [-h] [-c CONFIG] --model-path MODEL_PATH [-r RESUME] [-de DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  --model-path MODEL_PATH
                        path to model.pth to test (default: None
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -de DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
```

