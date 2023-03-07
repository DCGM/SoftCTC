# SoftCTC &ndash; Semi-Supervised Learning for Text Recognition using Soft Pseudo-Labels

This repository contains source codes for [SoftCTC](https://arxiv.org/abs/2212.02135) paper. 
In the `soft_ctc` subdirectory you can find:
- `models/`
  - `connections.py` - Model for storing transcription variants of single text line
  - `batch_connections.py` - Model for batched transcription variants
- `libs/` - Compiled libraries for CUDA and OpenCL
- `equations.py` - Implementation of SoftCTC loss equations as proposed in the paper
- `multi_ctc_loss.py` - Parallel MultiCTC loss implementation based on PyTorch CTC 
- `soft_ctc_loss.py` - SoftCTC loss implementation in PyTorch
- `soft_ctc_loss_cuda.py` - SoftCTC loss adapter for CUDA implementation
- `soft_ctc_loss_opencl.py` - SoftCTC loss adapter for OpenCL implementation

### CUDA and OpenCL

Current CUDA and OpenCL implementations are compiled for `Python 3.7-11` and `PyTorch 1.13.1`.
In case you need it for a different configuration, you can compile it on your own from the source codes in `soft_ctc_gpu_lib` or [SoftCTC-GPU](https://github.com/DCGM/SoftCTC-GPU) using the included instructions.

### Paper abstract
> This paper explores semi-supervised training for sequence tasks, such as Optical Character Recognition or Automatic Speech Recognition. We propose a novel loss function – SoftCTC – which is an extension of CTC allowing to consider multiple transcription variants at the same time. This allows to omit the confidence based filtering step which is otherwise a crucial component of pseudo-labeling approaches to semi-supervised learning. We demonstrate the effectiveness of our method on a challenging handwriting recognition task and conclude that SoftCTC matches the performance of a finely-tuned filtering based pipeline. We also evaluated SoftCTC in terms of computational efficiency, concluding that it is significantly more efficient than a naïve CTC-based approach for training on multiple transcription variants, and we make our GPU implementation public.

## Please cite

If you use SoftCTC, please cite:

> Kišš, M., Hradiš, M., Beneš, K., Buchal, P., & Kula, M. (2022). SoftCTC &ndash; Semi-Supervised Learning for Text Recognition using Soft Pseudo-Labels. arXiv preprint arXiv:2212.02135.

## Usage

### Import

[//]: # (To import SoftCTC loss, you can either import python/PyTorch implementation:)

```python
from soft_ctc.soft_ctc_loss import SoftCTCLoss
```

[//]: # (or you can import CUDA implementation for python 3.10:)

[//]: # (```python)

[//]: # (from soft_ctc.soft_ctc_loss_cuda import SoftCTCLoss)

[//]: # (```)

### Convert confusion network to `Connections` model

You can use `convert_confusion_network_to_connections` from `soft_ctc.models.connections` to convert confusion network (a list of dictionaries with characters as keys and probabilities as values) into a transition matrix (`Connections` model) and labels (ordered sequence of keys from the confusion network).

```python
# confusion_network = [{'C': 0.5, None: 0.5}, {'A': 0.7, 'U': 0.2, None: 0.1}, {'T': 1.0}, {'E': 0.6, 'S': 0.4}]
# character_set = ["<BLANK>"] + list("ACESTU")

from soft_ctc.models.connections import convert_confusion_network_to_connections, convert_characters_to_labels

blank = character_set.index("<BLANK>")
confusion_network = convert_characters_to_labels(confusion_network, character_set)
labels, connections = convert_confusion_network_to_connections(confusion_network, blank)
```

### Stack `Connections` and labels into `BatchConnections` and batched labels

To stack multiple transition matrices and labels into a batch structures, you can use `BatchConnections` and `stack_labels` from `soft_ctc.models.batch_connections`
```python
# all_connections: list of Connections in batch
# all_labels: list of labels in batch

from soft_ctc.models.batch_connections import BatchConnections, stack_labels, calculate_target_size

target_size = calculate_target_size([connections.size() for connections in all_connections])
batch_connections = BatchConnections.stack_connections(all_connections, target_size)
batch_labels = stack_labels(all_labels, blank, target_size)
```

### Calculate the SoftCTC loss
The initialization os the `SoftCTCLoss` has two optional parameters: `norm_step` and `zero_infinity`. The `norm_step` specifies how often the normalization is performed to prevent underflow (i.e. `norm_step=10` means that the normalization is done every 10-th frame in logits). If the `zero_infinity` flag is set, the individual losses which are equal to infinity are zeroed to prevent the training process from collapsing.

```python
# logits: raw network output of shape (N,C,T) where N is batch size, C is number of characters including blank, and T is 
# the number of frames (width) in logits

softctc = SoftCTCLoss(norm_step=10, zero_infinity=True)
loss = softctc(logits, batch_connections, batch_labels)

loss.mean().backward()
```

## Numerical stability
 Currently, the SoftCTC loss is calculated in probabilities and NOT log-probabilities.
 Due to this fact, the calculation may be mathematically unstable (underflow may occur), especially when the logits contain a very flat probability distribution at the beginning of the training process.
 Options to eliminate this are to use a smaller normalization step (`norm_step`), or to calculate in double precision.