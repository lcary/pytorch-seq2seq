pytorch-seq2seq
===============

A small python library with neural networks created from the following tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Requirements:
 - Python 3.7 (due to new-style type hints)
 - `pip install -r requirements.txt`

Usage
-----

From repo root:
```
python main.py
```

This will train a sequence-to-sequence neural network, and output several plots
as well as example sentence matrices for the trained network.

See `python main.py --help` for a full list of commandline arguments.

Unit tests
----------

Make sure to activate the virtual environment before running.

From repo root:
```bash
./runtests
```

Background
----------

A Recurrent Neural Network (RNN) operates on a sequence and uses its output
as input for subsequent steps.

A Sequence to Sequence (seq2seq) network, also known as an "Encoder Decoder"
network, is a model that consists of 2 RNNs. The first RNN is called the encoder,
which takes a sequence as input and outputs a vector. The vector encodes the
meaning of the input sequence into one vector in an N-dimensional space of
sentences. The second RNN is called the decoder, which reads the encoder's
output vector to produce an output sequence.

With a seq2seq network, each input sequence does not need to correspond to each
output. Sequence length and order can vary between inputs and output sequences.
For example, order and length vary in the input/output sequence of characters:

    "Je ne suis pas le chat noir" -> “I am not the black cat”
