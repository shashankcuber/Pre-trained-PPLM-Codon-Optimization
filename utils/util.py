import random
from CAI import CAI, relative_adaptiveness, RSCU
from utils.RNAdegformer.stability_prediction_util import *
import torch

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
        
        # print(f"Padding removed: for id {id}", len(seq)-i+1)        
        # remove padding from predicted sequence
        pred_logits_without_pad.append(output_seq_logits[id][ :len(seq)-cnt_zero+1, : ])
    
    return pred_logits_without_pad

def convert_index_to_codons(seq_pad_removed, cds_token_dict):
    # print("seq_pad_removed: ", len(seq_pad_removed))
    codon_seq = ""
    #converting the codon to index dictionary to index to codon dictionary for inference of output sequences
    index_to_word = {v: k for k, v in cds_token_dict.items()}

    for index in seq_pad_removed:
        # index is of type tensor so need to convert to int
        # print(type(index))
        if int(index) == -100:
            codon_seq += index_to_word[random.choice([i for i in range(1, len(index_to_word))])]
        else:
            codon_seq += index_to_word[int(index)]
    # print("codon_seq: ", len(codon_seq))
    return codon_seq

def get_batch_gc_content(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, test=False):
    output_batch_gc = []
    target_batch_gc = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
        # print("trimmed_output_seq: ", trimmed_output_seq, '\n', len(trimmed_output_seq))
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
        # print("predicted_seq: ", predicted_seq, '\n', len(predicted_seq))
        gc = (predicted_seq.count('g') + predicted_seq.count('c')) / len(predicted_seq)
        # print("GC content: ", gc)
        output_batch_gc.append(gc)
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
            gc_gt = (target_seq.count('g') + target_seq.count('c')) / len(target_seq)
            # print("GC content GT: ", gc_gt)
            target_batch_gc.append(gc_gt)
    return torch.tensor(output_batch_gc), torch.tensor(target_batch_gc)

def get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, ref_seqs, test=False):
    output_batch_cai = []
    target_batch_cai = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    # print("predicted_output_logits: ", predicted_output_logits.shape)
    # print("cds_data_sorted: ", cds_data_sorted.shape)
    weights = relative_adaptiveness(ref_seqs)
    # print("weights: ", weights)
    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
        # print("trimmed_output_seq: ", trimmed_output_seq, '\n', len(trimmed_output_seq))
        
        
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
        # print("Predicted Codon sequence: ", predicted_seq, '\n', len(predicted_seq))
        # print("predicted_seq: ", predicted_seq, '\n', len(predicted_seq))
        # target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
        
        # print("predicted_seq: ", predicted_seq)
        # print("target_seq: ", target_seq)
        # cai_pd = CAI(predicted_seq, weights=weights)
        # print(f"Predicted Moderna CAI = {cai_pd:.4f}")
        # print(f"Predicted Pfizer CAI = {cai_pd:.4f}")
        # print(f"Predicted CureVac CAI = {cai_pd:.4f}")
        # print("Predicted Pfizer CAI : ", CAI(predicted_seq, weights=weights), '|', "Original -CAI: ", CAI(target_seq, weights=weights))
        
        output_batch_cai.append(CAI(predicted_seq, weights=weights))
        # target_batch_cai.append(CAI(target_seq, weights=weights))
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
            target_batch_cai.append(CAI(target_seq, weights=weights))
    
    # print(output_batch_cai)
    # print(target_batch_cai)
    # print('\n')
    return torch.tensor(output_batch_cai, requires_grad=True), torch.tensor(target_batch_cai), predicted_seq

def get_batch_stability(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, temperature, package, stability_type, test=False):
    output_batch_stability = []
    predicted_seq_stability = []
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    T = [temperature]

    for i in range(len(seq_lens)):
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
        # print("trimmed_output_seq: ", trimmed_output_seq, '\n', len(trimmed_output_seq))
        predicted_seq = convert_index_to_codons(trimmed_output_seq, cds_token_dict)
        target_seq = " "
        if test == True:
            trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
            target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
        # trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
        # target_seq = convert_index_to_codons(trimmed_target_seq, cds_token_dict)
        # print("predicted_seq: ", predicted_seq, '\n', len(predicted_seq))
        sp = StabilityPredictor(predicted_seq, package, T)
        # sp_pred = StabilityPredictor(target_seq, package, T)
        if stability_type == 'deg':
            deg = sp.get_stability_from_rnadegformer()[0]
            # deg_gt = sp_pred.get_stability_from_rnadegformer()[0]
            # print("deg: ", deg, "|", "GT_deg: ", deg_gt)
            output_batch_stability.append(deg)
        else:
            if package == "vienna":
                RNA.cvar.temperature = temperature
                fold_compound = RNA.fold_compound(predicted_seq)
                structure, mfe = fold_compound.mfe()
                # print(f"Moderna MFE Predicted = {mfe:.4f} kcal/mol")
                # print(f"Pfizer MFE Predicted = {mfe:.4f} kcal/mol")
                # print(f"CureVac MFE Predicted = {mfe:.4f} kcal/mol")
                
                output_batch_stability.append(mfe)
                # output_batch_stability.append(mfe/len(predicted_seq))

                if test == True:
                    fold_compound = RNA.fold_compound(target_seq)
                    pred_s , pred_mfe = fold_compound.mfe()
                
                    predicted_seq_stability.append(pred_mfe)
                # predicted_seq_stability.append(pred_mfe/len(target_seq))

            else:
            # print("mfe: ", mfe)
                output_batch_stability.append(mfe)
               
            # mfe = sp.get_all_features()[3]
            # # print("mfe: ", mfe)
            # output_batch_stability.append(mfe)
        # print("predicted_seq: ", predicted_seq)

    return torch.tensor(output_batch_stability), torch.tensor(predicted_seq_stability)


        


    
