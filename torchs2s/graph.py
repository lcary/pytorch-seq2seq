from typing import List

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
from torch import Tensor


def save_plot(points: List[float]) -> None:
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    filename = 'plot.png'
    fig.savefig(filename)
    print('Saved {}'.format(filename))


def save_attention_matrix(input_sentence: str,
                          output_words: List[str],
                          attentions: Tensor,
                          filename: str) -> None:
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(filename)
    print('Saved {}'.format(filename))
