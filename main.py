from transformer import (
    InputEmbeddings
)
import torch

def main():
    embedding_layer = InputEmbeddings(
        vocab_size=10_000,
        d_model=512
    )

    embedding_output = embedding_layer(torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7 , 8]]
    ))
    print(embedding_output)

if __name__ == "__main__":
    main()