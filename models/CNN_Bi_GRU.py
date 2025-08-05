import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS

class CNNBiGRUModel(nn.Module):
    def __init__(self, num_amino_acids=21, num_codons=62, embedding_dim=21, hidden_dim=128, num_layers=4, dropout_rate=0.5):
        super(CNNBiGRUModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=21, padding_idx=0)
        
        # CNN layer
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # Bidirectional GRU Layer
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layers
        self.fc1 = nn.Linear(2*128, 128) # 2*128 because of bidirectional
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 62) # 64 because we want to predict 64 codons and +1 for padding index when using pack padding we don't need one extra
        self.token_amino = {v: k for k, v in AMINO_ACID_DICT.items()}
        self.token_codon = {v: k for k, v in CODON_DICT.items()}
        
        # Masking function remains unchanged
    

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


    def forward(self, x, seq_lens):
        # amino_seq = x
        # Embedding layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x= x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len] for CNN

        # CNN layer
        cnn_out = self.relu(self.conv1(x))  # [batch_size, hidden_dim, seq_len]
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]

        # Packing sequence for RNN
        packed_input = pack_padded_sequence(cnn_out, seq_lens, batch_first=True)
        
        # GRU layer
        packed_output, _ = self.bi_gru(packed_input)
        
        # Unpacking sequence for GRU
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

       
        # output = self.dropout(output)
        output= self.fc1(output.contiguous().view(-1, output.shape[2]))
        output= self.bn(output)
        output= self.relu(output)
        output= self.fc2(output)
        output= output.view(x.size(0), -1 , 62)
        output= F.softmax(output, dim=-1)
        # print(output[1][4:5][:])
        # masked_logits = self.mask_logits(output, seq_lens, amino_seq=amino_seq)
        # print(masked_logits[1][4:5][:])
        # masked_logits = F.softmax(masked_logits, dim=-1)
        # print(masked_logits[1][4:5][:])
        return output
        # return masked_logits

# # Instantiate the model with specific parameters
# model = CNNBiGRUModel(
#     num_amino_acids=22, 
#     num_codons=65, 
#     embedding_dim=128, 
#     hidden_dim=256, 
#     num_layers=2, 
#     dropout_rate=0.5
# )
