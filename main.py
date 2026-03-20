from transformer import (
    InputEmbeddings,
    PositionalEncoding
)
import torch

VOCAB_SIZE =        10_000
D_MODEL =           512
MAX_SEQ_LENGTH =    4

def main():


    embedding_layer = InputEmbeddings(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL
    )

    embedded_output = embedding_layer(torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7 , 8]]
    ))
    print(embedded_output.shape)

    pos_encoding_layer = PositionalEncoding(d_model=D_MODEL, max_seq_length=MAX_SEQ_LENGTH)
    pos_encoded_output = pos_encoding_layer(embedded_output)
    print(pos_encoded_output.shape)

if __name__ == "__main__":
    main()