import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS

class CodonPredictionModel(nn.Module):
    def __init__(self, bert_model):
        super(CodonPredictionModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 61)  # 60 codons + 1 for padding
        self.token_amino = {v: k for k, v in AMINO_ACID_DICT.items()}
        self.token_codon = {v: k for k, v in CODON_DICT.items()}


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
                # print(seq_lens[batch_idx])
                # print(len(amino_seq[batch_idx]))
                # Get the amino acid at the current position
                amino_acid_idx = amino_seq[batch_idx, pos_idx].item()
                # print(amino_acid_idx)
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
               

    def forward(self, x, seq_lens, attention_mask, mask):
        amino_seq = x.clone()
        # print("bf")
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        # print("af")
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        output= logits.view(x.size(0), -1 , 61)
        # print(logits.shape)
        # print(logits)
        # return logits
        if mask is True:
            masked_logits = self.mask_logits(output, seq_lens, amino_seq=amino_seq)
            # print("Masking Done")
            return nn.functional.softmax(masked_logits, dim=-1)
        return nn.functional.softmax(logits, dim=-1)