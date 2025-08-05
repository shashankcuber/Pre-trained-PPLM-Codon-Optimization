import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS
# Define the PyTorch model class
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel,self).__init__()
        #embedding dim changed from 64 to 61
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=128, padding_idx=0) # 64->62->21
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2, #changed from 2-8
            dropout=0.3, # changef from 0.5 -> 0.1
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(2*256, 128) # 2*128 because of bidirectional
        self.bn1 = nn.BatchNorm1d(128)
        # self.tanh = nn.ReLU()
        #tanh 
        self.tanh_1 = nn.LeakyReLU()
        self.tanh_2 = nn.LeakyReLU()
        self.tanh_3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 256) # 64 because we want to predict 64 codons and +1 for padding index when using pack padding we don't need one extra
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 61)
        self.token_amino = {v: k for k, v in AMINO_ACID_DICT.items()}
        self.token_codon = {v: k for k, v in CODON_DICT.items()}
        # self.softmax = nn.softmax(dim=2) #We don't need as cross entropy loss has softmax inbuilt it uses log_softmax and nll_loss
    

    def mask_logits(self, logits, seq_lens, amino_seq):
        # mask = torch.full_like(logits, -1e7)  # Create a mask filled with -1e7

        # for i in range(logits.size(0)):
        #     valid_codon_indices = [CODON_DICT[codon.lower()] for aa in amino_seq[i][:seq_lens[i]] for codon in SYNONYMOUS_CODONS[self.token_amino[aa.item()]]]
        #     mask[i, :seq_lens[i], valid_codon_indices] = 0  # Only valid codons get a zero mask

        # return logits + mask  # Add the mask to the logits; invalid logits get reduced to a very low value
        
        batch_size = logits.size(0)
        mask = torch.full_like(logits, -1e9)
        # print(mask[2, 4:5, :])
        # Iterate over each example in the batch
        for batch_idx in range(batch_size):
            # Iterate over each position in the sequence
            for pos_idx in range(seq_lens[batch_idx]):
                # Get the amino acid at the current position
                amino_acid_idx = amino_seq[batch_idx, pos_idx].item()
                amino_acid = self.token_amino[amino_acid_idx]
                # Get the list of valid codon indices for this amino acid
                valid_codons = SYNONYMOUS_CODONS[amino_acid]
                valid_codon_indices = [CODON_DICT[codon.lower()] for codon in valid_codons]
                # print(valid_codon_indices)
                # Set the mask to 0 (unmask) at the positions of valid codons
                mask[batch_idx, pos_idx, valid_codon_indices] = 0

        # print(mask[2, 4:5, :])
        # Apply the mask to the logits
        masked_logits = logits + mask
        return masked_logits
               

    def forward(self, x, seq_lens, mask):
        amino_seq = x
        x = self.embedding(x)
        packed_emb = pack_padded_sequence(x, seq_lens, batch_first=True)
        packed_output, _  = self.bi_gru(packed_emb)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # print(x[1][:][:])
        """
        Output here looks like this:
        It handles padding region for max len in current batch by keeping them zero
        tensor([[-0.0360,  0.0393, -0.0616,  ..., -0.0695, -0.0450,  0.0601],
        [-0.0536,  0.0471, -0.0966,  ..., -0.0721, -0.0457,  0.0844],
        [-0.0645,  0.0475, -0.1171,  ..., -0.0744, -0.0433,  0.1026],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
        grad_fn=<SliceBackward0>)

        """
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.tanh(x)
        # x = self.fc2(x)
        # x = self.softmax(x)

        output= self.fc1(output.contiguous().view(-1, output.shape[2]))
        output= self.bn1(output)
        output= self.tanh_1(output)
        output= self.fc2(output)
        output= self.bn2(output)
        output= self.tanh_2(output)
        output= self.fc3(output)
        output= self.bn3(output)
        output= self.tanh_3(output)
        output= self.fc4(output)
        output= output.view(x.size(0), -1 , 61)
        output= F.softmax(output, dim=-1)
        # print(output[1][4:5][:])
        if mask=="True":
            masked_logits = self.mask_logits(output, seq_lens, amino_seq=amino_seq)
            # print("Masking Done")
            return masked_logits
        # print(masked_logits[1][4:5][:])
        # masked_logits = F.softmax(masked_logits, dim=-1)
        # print(masked_logits[1][4:5][:])
        # exit()
        return output
