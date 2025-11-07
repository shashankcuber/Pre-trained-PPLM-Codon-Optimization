import random
from cai2 import CAI
import torch
import RNA
from data_preprocessing_protbert import read_data_from_file
def remove_pad_from_output(output_seq_logits, cds_data):
    pred_logits_without_pad = []
    for id, seq in enumerate(cds_data):
        cnt_zero = 0
         #check post 0's in cds_data
        i = len(cds_data[id])-1

        while i>0:
            if seq[i] == 0:
                cnt_zero +=1
            else:
                break
            i-=1
             
        # remove padding from predicted sequence
        pred_logits_without_pad.append(output_seq_logits[id][ :len(seq)-cnt_zero+1, : ])
    
    return pred_logits_without_pad

def convert_index_to_codons(seq_pad_removed, cds_token_dict):
    codon_seq = ""
    #converting the codon to index dictionary to index to codon dictionary for inference of output sequences
    index_to_word = {v: k for k, v in cds_token_dict.items()}

    for index in seq_pad_removed:
        # index is of type tensor so need to convert to int
        if int(index) == -100:
            codon_seq += index_to_word[random.choice([i for i in range(1, len(index_to_word))])]
        else:
            codon_seq += index_to_word[int(index)]
    return codon_seq

def get_batch_gc_content(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, test=False):
    output_batch_gc = []
    target_batch_gc = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
        
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
      
        gc = (predicted_seq.count('g') + predicted_seq.count('c')) / len(predicted_seq)
       
        output_batch_gc.append(gc)
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
            gc_gt = (target_seq.count('g') + target_seq.count('c')) / len(target_seq)
            
            target_batch_gc.append(gc_gt)
    return torch.tensor(output_batch_gc), torch.tensor(target_batch_gc)

def get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, host_organism, test=False):
    output_batch_cai = []
    target_batch_cai = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
   
    # weights = relative_adaptiveness(ref_seqs)
    if host_organism == 'human':
        weights = read_data_from_file('./codon_frequency_table/human_relative_adaptiveness.json')
    elif host_organism == 'ecoli':
        weights = read_data_from_file('./codon_frequency_table/ecoli_relative_adaptiveness.json')
    elif host_organism == 'chinese-hamster':
        weights = read_data_from_file('./codon_frequency_table/chinese_hamster_relative_adaptiveness.json')

    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
       
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
        
        output_batch_cai.append(CAI(predicted_seq, weights=weights))
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
            target_batch_cai.append(CAI(target_seq, weights=weights))
    
    return torch.tensor(output_batch_cai, requires_grad=True), torch.tensor(target_batch_cai), predicted_seq

def get_batch_stability(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, temperature, package, stability_type, test=False):
    output_batch_stability = []
    predicted_seq_stability = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    T = [temperature]

    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
       
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
        target_seq = " "
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)

        if stability_type == 'deg':
            '''
            Not functional currently as RNAdegformer package is not available
            '''
            # deg = sp.get_stability_from_rnadegformer()[0]
            # deg_gt = sp_pred.get_stability_from_rnadegformer()[0]
            # print("deg: ", deg, "|", "GT_deg: ", deg_gt)
            deg = 0
            output_batch_stability.append(deg)
        else:
            if package == "vienna":
                RNA.cvar.temperature = temperature
                fold_compound = RNA.fold_compound(predicted_seq)
                structure, mfe = fold_compound.mfe()
                
                output_batch_stability.append(mfe)

                if test == True:
                    fold_compound = RNA.fold_compound(target_seq)
                    pred_s , pred_mfe = fold_compound.mfe()
                
                    predicted_seq_stability.append(pred_mfe)

            else:
                output_batch_stability.append(mfe)
               

    return torch.tensor(output_batch_stability), torch.tensor(predicted_seq_stability)


        


    
