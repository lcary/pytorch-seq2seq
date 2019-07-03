import torch.optim as optim

from torchs2s.constants import DEVICE
from torchs2s.language import Language
from torchs2s.networks import EncoderRNN, AttentionDecoderRNN, NetworkContext


def get_network_context(input_size=1000, output_size=1000):
    encoder = EncoderRNN(input_size, 256).to(DEVICE)
    decoder = AttentionDecoderRNN(256, output_size, dropout_p=0.1).to(DEVICE)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
    return NetworkContext(
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        Language("eng"),
        Language("fra")
    )
