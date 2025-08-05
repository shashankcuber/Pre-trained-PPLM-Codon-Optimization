import torch
import torch.nn as nn
import random
from utils.CONSTANTS import SYNONYMOUS_CODONS
from collections import defaultdict
from CAI import CAI, relative_adaptiveness, RSCU
# from statistics import geometric_mean


"""
Cross Entropy Loss
"""

class CrossEntropyLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction = 'mean',ignore_index=-100)
        self.device = device
    
    def forward(self, predicted, target):
        return self.loss(predicted, target).to(self.device)
    
"""
MSE loss for CAI
"""

class CaiMseLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, y_pred, y):
        y_pred = y_pred.to(self.device)
        return self.loss(y_pred, y).to(self.device)


class StabilityMseLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss = nn.MSELoss(reduction='sum')
        # self.loss = nn.MSELoss(reduction='mean')

    
    def forward(self, y_pred, y):
        y_pred = y_pred.to(self.device)
        return self.loss(y_pred, y).to(self.device)

"""
Using CAI library it does not work well.
"""
class CAI_LOSS(nn.Module):
    def __init__(self, cds_token_dict, ref_seqs):
        super().__init__()
        self.token_cds_dict = {v:k for k,v in cds_token_dict.items()}
        print("TOKEN TO CODON DICT--->", self.token_cds_dict, len(self.token_cds_dict))
        assert len(self.token_cds_dict) == len(cds_token_dict)
        self.ref_seq = ref_seqs
        self.weights = relative_adaptiveness(self.ref_seq)
        self.rscu = RSCU(self.ref_seq)
    
    def get_predicted_codon_seq(self, predicted_logit_trimmed):
        
        seq = ""
        for index in predicted_logit_trimmed:
            # index is of type tensor so need to convert to int
            # print(type(index))
            if int(index) == 0:
                seq += self.token_cds_dict[random.choice([i for i in range(1, len(self.token_cds_dict))])]
            else:
                seq += self.token_cds_dict[int(index)]
        
        return seq
    
    def forward(self, output_seq_logits, seq_lens, device):
        # print("RSCU--->", self.rscu, len(self.rscu))
        # print("\n")
        # print("Relative Adaptiveness--->", self.weights, len(self.weights))
        loss = 0
        loss_cai = 0
        for i in range(0, len(output_seq_logits)):
            #get the maximum index from dim=-1 of output_seq_logits
            predicted_tokens = torch.argmax(output_seq_logits[i], dim=-1).to(device)
            # print("Predicted tokens--->", predicted_tokens.shape)
            predicted_tokens_trimmed = predicted_tokens[:seq_lens[i]].to(device)
            # print("Predicted tokens trimmed--->", predicted_tokens_trimmed.shape)
            predicted_codon_seq = self.get_predicted_codon_seq(predicted_tokens_trimmed)
            # print("Predicted codon seq--->", len(predicted_codon_seq), predicted_codon_seq)
            # print("\n")
            cai_loss = torch.tensor(CAI(predicted_codon_seq, weights=self.weights))
            loss = loss + torch.log(cai_loss)
        
        loss = torch.divide(loss, len(output_seq_logits)).requires_grad_(True)
        return loss
        
       

"""
Implementation from Scratch Differentiable CAI Loss
""" 

class SoftmaxCAI(nn.Module):
    def __init__(self, cds_token_dict, ref_seqs):
        super().__init__()
        self.token_cds_dict = {v:k for k,v in cds_token_dict.items()}
        print("TOKEN TO CODON DICT--->", self.token_cds_dict, len(self.token_cds_dict))
        assert len(self.token_cds_dict) == len(cds_token_dict)
        self.ref_seqs = ref_seqs
        self.rscu = self.RSCU_calculation(self.ref_seqs)
        # self.rscu = RSCU(self.ref_seqs)

    def RSCU_calculation(self, ref_seqs):

        codon_count = defaultdict(int)

        for seq in ref_seqs:
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                codon_count[codon] += 1
        
        print("Codon count--->", codon_count, len(codon_count))
        rscu = defaultdict(float)

        for _, codons in SYNONYMOUS_CODONS.items():
            
            
            total_freq_synonymous_codons = 0

            for codon in codons:
                codon = codon.lower()
                total_freq_synonymous_codons = total_freq_synonymous_codons + codon_count[codon]
            
            #denominator of RSCU
            average_total_codon_freq = total_freq_synonymous_codons / len(codons)

            for codon in codons:
                codon = codon.lower()
                rscu[codon] = codon_count[codon] / average_total_codon_freq
            
        return rscu
    
    def get_predicted_codon_seq(self, predicted_logit_trimmed):
        
        seq = ""
        for index in predicted_logit_trimmed:
            # index is of type tensor so need to convert to int
            # print(type(index))
            if int(index) == 0:
                seq += self.token_cds_dict[random.choice([i for i in range(1, len(self.token_cds_dict))])]
            else:
                seq += self.token_cds_dict[int(index)]
        
        return seq
    
    def get_softmax_weight_for_codon(self, codon):

        if codon in self.rscu:
            # numerator = torch.exp(torch.tensor(self.rscu[codon]))
            # denominator = 0
            
            # for _, codon_list in SYNONYMOUS_CODONS.items():
            #     codon_list = [c.lower() for c in codon_list]

            #     if codon in codon_list:
            #         for c in codon_list:
            #             denominator = denominator + torch.exp(torch.tensor(self.rscu[c]))
            #             # print(f'Codon {c}--->', torch.exp(torch.tensor(self.rscu[c])))
            # # print("Numerator--->", numerator, "Denominator--->", denominator)
            # if denominator == 0:
            #     print("Error: Denominator is zero for codon ", codon, " in RSCU")
            #     return 1e7
            # return torch.divide(numerator, denominator)
            synonymous_codons = []
            for _ , codons in SYNONYMOUS_CODONS.items():
                if codon.upper() in codons:
                    synonymous_codons = codons
                    break

            rscu_values_synonymous = [self.rscu[c.lower()] for c in synonymous_codons]
            
            softmax_rscu_synonymous = torch.nn.functional.softmax(torch.tensor(rscu_values_synonymous, dtype=torch.float, requires_grad=True), dim=0)
            
            codon_index = synonymous_codons.index(codon.upper())
            wi = softmax_rscu_synonymous[codon_index]
            return wi

        else:
            print("Error: Codon not found in RSCU")
            return 1e7

    def geometric_mean(self, weights):
        """
        Computes the geometric mean of a list of tensors in a PyTorch-friendly way.
        Args:
            weights (list of torch.Tensor): The input list of tensors.
        Returns:
            torch.Tensor: The geometric mean of the input tensors.
        """

        # Convert the list of tensors to a single tensor
        input_tensor = torch.stack(weights)

        # Ensure the input tensor doesn't contain negative or zero values
        assert torch.all(input_tensor > 0), "All tensor values should be positive for the geometric mean calculation."

        # Compute the log geometric mean
        log_mean = torch.mean(torch.log(input_tensor))

        # Return the geometric mean
        return torch.exp(log_mean)

    def forward(self, output_seq_logits, seq_lens, device):

        # print("RSCU--->", self.rscu, len(self.rscu))
        # print("\n")
        
        batch_cai = 0

        for i in range(0, len(output_seq_logits)):
            #get the maximum index from dim=-1 of output_seq_logits
            predicted_tokens = torch.argmax(output_seq_logits[i], dim=-1).to(device)
            # print("Predicted tokens--->", predicted_tokens.shape)
            predicted_tokens_trimmed = predicted_tokens[:seq_lens[i]].to(device)
            # print("Predicted tokens trimmed--->", predicted_tokens_trimmed.shape)
            predicted_codon_seq = self.get_predicted_codon_seq(predicted_tokens_trimmed)
            # print("Predicted codon seq--->", len(predicted_codon_seq), predicted_codon_seq)
            # print("\n")
            weights = []

            for j in range(0, len(predicted_codon_seq), 3):
                codon = predicted_codon_seq[j:j+3]
                codon_relaitve_adaptiveness = self.get_softmax_weight_for_codon(codon)
                weights.append(codon_relaitve_adaptiveness)
                # print("Codon--->", codon, "Codon relative adaptiveness--->", codon_relaitve_adaptiveness)
            
            # print("Weights--->", weights, len(weights))
            # print("\n")

            # calculating the CAI by using geometric mean
            # weights_product = 1
            # for weight in weights:
            #     weights_product = weights_product * weight 

            # cai = torch.pow(weights_product, 1/len(weights))

            """ 
            Taking log of CAI to make it additive and easy to differentiate as well as to avoid vanishing CAI due to 
            multiplication of small numbers
            """
            cai = self.geometric_mean(weights)
            # print("Batch CAI--->", cai)
            # cai = 
            batch_cai += cai
        
        batch_cai = batch_cai / len(output_seq_logits)

        return batch_cai

# class SoftmaxCAI(nn.Module):
#     def __init__(self, cds_token_dict, ref_seqs, device):
#         super().__init__()
#         self.token_cds_dict = {v: k for k, v in cds_token_dict.items()}
#         self.ref_seqs = ref_seqs
#         self.device = device
#         self.rscu = self.RSCU_calculation(self.ref_seqs)

#     def RSCU_calculation(self, ref_seqs):
#         codon_count = defaultdict(int)
#         for seq in ref_seqs:
#             for i in range(0, len(seq), 3):
#                 codon = seq[i:i+3].lower()
#                 codon_count[codon] += 1
        
#         rscu = defaultdict(float)
#         for _, codons in SYNONYMOUS_CODONS.items():
#             total = sum(codon_count[codon.lower()] for codon in codons)
#             for codon in codons:
#                 codon = codon.lower()
#                 if total > 0:  # Avoid division by zero
#                     rscu[codon] = codon_count[codon] / (total / len(codons))
#                 else:
#                     rscu[codon] = 0.0  # Or handle this however you see fit
#         return {k: torch.tensor(v, dtype=torch.float, requires_grad=False).to(self.device) for k, v in rscu.items()}

#     def get_softmax_weight_for_codon(self, codon):
#         synonymous_codons = [c for c_list in SYNONYMOUS_CODONS.values() if codon.upper() in c_list for c in c_list]
#         rscu_values = torch.stack([self.rscu[c.lower()] for c in synonymous_codons]).to(self.device)
#         softmax_rscu = torch.softmax(rscu_values, dim=0)
#         return softmax_rscu[synonymous_codons.index(codon.upper())]

#     def geometric_mean(self, weights):
#         weights = torch.stack(weights).to(self.device)
#         weights = torch.clamp(weights, min=1e-8)  # Avoid log(0)
#         log_mean = torch.mean(torch.log(weights))
#         return torch.exp(log_mean)
    
#     def get_predicted_codon_seq(self, predicted_logit_trimmed):
#         predicted_codon_seq = []
#         for index in predicted_logit_trimmed:
#             if int(index) == 0:
#                 #choose a random integer between 1-61
#                 random_index = random.choice([i for i in range(1, len(self.token_cds_dict))])
#                 predicted_codon_seq.append(self.token_cds_dict[random_index])
#             predicted_codon_seq.append(self.token_cds_dict[int(index)])
        
#         return predicted_codon_seq

#     def forward(self, output_seq_logits, seq_lens):
#         batch_cai_scores = []

#         for i, logits in enumerate(output_seq_logits):
#             predicted_tokens = torch.argmax(logits, dim=-1).to(self.device)
#             predicted_tokens_trimmed = predicted_tokens[:seq_lens[i]]
#             predicted_codon_seq = [self.token_cds_dict[int(index)] for index in predicted_tokens_trimmed]
#             # predicted_codon_seq = self.get_predicted_codon_seq(predicted_tokens_trimmed)

#             weights = [self.get_softmax_weight_for_codon(codon) for codon in predicted_codon_seq]

#             cai_score = self.geometric_mean(weights)
#             # print("CAI score for a sequence in batch--->", cai_score)
#             batch_cai_scores.append(cai_score)

#         batch_cai = torch.stack(batch_cai_scores).mean().to(self.device)

#         return batch_cai 