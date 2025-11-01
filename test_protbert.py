import argparse
import torch
from tqdm.auto import tqdm as tqdm
from transformers import BertModel
from models.PROTBERT import CodonPredictionModel
from data_preprocessing_protbert import start_preprocessing_probert
from utils.util import get_batch_cai, get_batch_stability, get_batch_gc_content
import gradio as gr

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_pad_trimmed_cds_data(cds_data, max_seq_len):
    cds_data_trimmed = []
    for id, seq in enumerate(cds_data):
        # print(type(seq))
        cds_data_trimmed.append(seq[0:max_seq_len])
    
    cds_data_trimmed = torch.stack(cds_data_trimmed)
    return cds_data_trimmed

def get_padded_output(output_seq_logits, seq_lens):
    
    for i in range(0, len(seq_lens)):
        output_seq_logits[i][seq_lens[i]:][:] = -100
    
    return output_seq_logits
    
def get_correct_tags(output_seq_logits, cds_data, seq_lens):
    right_token =0
    total_tokens = 0
    for i in range(0, len(seq_lens)):
        seq_i = torch.argmax(output_seq_logits[i], dim=-1)
        # Excluding Start and Stop codons
        seq_i = seq_i[:seq_lens[i]]
        cds_data_i = cds_data[i][:seq_lens[i]]
        total_tokens += len(cds_data_i)
        # get token wise accuracy
        for j in range(0, len(seq_i)):
            if seq_i[j] == cds_data_i[j]:
                right_token += 1
    
    return right_token, total_tokens


def test(model, test_loader, cai_type, mask, cds_token_dict, ref_seq_cds, max_seq_len, tool_pkg, temperature, stability_type):
    
        model.eval()
        right_tags_test = 0
        total_tags_test = 0
    
        total_cai = 0
        total_stb = 0
        total_gc = 0
        total_cai_original = 0
        cai_original_list = []
        cai_pred_list = []
        total_stb_original = 0
        mfe_original_list = []
        mfe_pred_list = []
        total_gc_original = 0
        gc_original_list = []
        gc_pred_list = []

        with torch.no_grad():
            for i,(batch) in enumerate(tqdm(test_loader)):
                aa_data = batch['input_ids']
                attention_mask = batch['attention_mask'].to(device)
                cds_data = batch['labels']
                seq_lens = torch.sum(cds_data != -100, dim=1)
                seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
                max_seq_len = max(seq_lens)

                aa_data_sorted = []
                cds_data_sorted = []
                for i in range(0, len(sorted_index)):
                    aa_data_sorted.append(aa_data[sorted_index[i]])
                    cds_data_sorted.append(cds_data[sorted_index[i]])
                
                aa_data_sorted = torch.stack(aa_data_sorted)
               
                cds_data_sorted = torch.stack(cds_data_sorted)
        
                aa_data_sorted = aa_data_sorted.to(device)
                cds_data_sorted = cds_data_sorted.to(device)

                #Predict the output sequence
                output_seq_logits = model(aa_data_sorted, seq_lens, attention_mask, mask=mask)
               
                # Trim padding from cds data to match output_seq_logits dimensions
                cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len)
                    
                if cai_type == 'mse' and (stability_type=='deg' or stability_type=='mfe'):
                    cai_pred, cai_target, predicted_orf = get_batch_cai(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, ref_seq_cds, test=True)
                    cai_pred_list += list(cai_pred)
                    cai_original_list += list(cai_target)
                   
                    total_cai += torch.sum(cai_pred)
                    total_cai_original += torch.sum(cai_target)

                    gc_pred, gc_target = get_batch_gc_content(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, test=True)
                    gc_pred_list += list(gc_pred)
                    gc_original_list += list(gc_target)
                    total_gc += torch.sum(gc_pred)
                    total_gc_original += torch.sum(gc_target)

                    stability_pred, stability_target = get_batch_stability(output_seq_logits, cds_data_sorted, cds_token_dict, seq_lens, temperature, tool_pkg, stability_type, test=True)
                    mfe_original_list += list(stability_target)
                    mfe_pred_list += list(stability_pred)
                   
                    total_stb += torch.sum(stability_pred)
                    total_stb_original += torch.sum(stability_target)
                right_tokens, total_tokens = get_correct_tags(output_seq_logits, cds_pad_trimmed, seq_lens)
                right_tags_test += right_tokens
                total_tags_test += total_tokens

        avg_cai = total_cai / len(test_loader.dataset)
        avg_stb = total_stb / len(test_loader.dataset)
        avg_gc = total_gc / len(test_loader.dataset)
        avg_gc *= 100  # Convert to percentage
        avg_cai_original = total_cai_original / len(test_loader.dataset)
        avg_stb_original = total_stb_original / len(test_loader.dataset)
        avg_gc_original = total_gc_original / len(test_loader.dataset)
        avg_gc_original *= 100  # Convert to percentage
       
    
        print(f'Test CAI = {avg_cai:.4f} | Test CAI Original = {avg_cai_original:.4f} | Test Stability Original = {avg_stb_original:.4f} | Test Stability = {avg_stb:.4f} | Test GC = {avg_gc:.4f} | Test GC Original = {avg_gc_original:.4f}')
        results =  {
            'Optimized ORF': predicted_orf,
            'CAI Optimized ORF': f'{avg_cai:.4f}',
            'CAI Wild Type': f'{avg_cai_original:.4f}',
            'Stability of Optimized ORF': f'{avg_stb:.4f}',
            'Stability of Wild Type': f'{avg_stb_original:.4f}',
            'GC Content of Optimized ORF': f'{avg_gc:.4f}',
            'GC Content of Wild Type': f'{avg_gc_original:.4f}'
            }
        return results

def run_inference(mask=True, stability_type='mfe', tool_pkg='vienna', temperature=37, host_organism='human', protein_seq='sars_cov2', model_type='human-long'):
    
    cai_type = 'mse'
    mask = mask
    tool_pkg = tool_pkg
    host_organism = host_organism
    stability_type = 'mfe'
    temperature = temperature
    protein_seq = protein_seq
    model_type = model_type


    if host_organism == 'human':
        dataset_path = './ref_set_sequences/human/hg19.json'
    elif host_organism == 'ecoli':
        dataset_path = './ref_set_sequence/ecoli/ecoli.json'
    elif host_organism == 'chinese-hamster':
        dataset_path = './ref_set_sequence/chinese_hamster/chinese_hamster.json'
    

    _, _ , test_loader, ref_seq_cds, cds_token_dict, max_seq_len  = start_preprocessing_probert(dataset_path, protein_seq, host_organism)
   
    if model_type == 'human':
        bert_model_path = './pretrained_models/human/ppLMCO-human.pt'
    elif model_type == 'ecoli':
        bert_model_path = './pretrained_models/ecoli/ppLMCO-ecoli.pt'
    elif model_type == 'chinese-hamster':
        bert_model_path = './pretrained_models/chinese_hamster/ppLMCO-ch.pt'

    # model_path = f'./saved_best_model/{bert_model_path}'
    # print(f"Model path: {model_path}")
    
    # Loding the ProtBert model as per the model type
    bert_model = BertModel.from_pretrained('Rostlab/prot_bert')
        
    # Freeze ProtBert parameters
    for param in bert_model.parameters():
        param.requires_grad = False

    model = CodonPredictionModel(bert_model).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters of PPLMCO =: {total_params}')
   
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Trainable parameters of PPLMCO =: {train_params}')
    print(f"Model testing - {bert_model_path} ")
   
    checkpoint = torch.load(bert_model_path, map_location=device)
    model = CodonPredictionModel(bert_model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = test(model, test_loader, cai_type, mask, cds_token_dict, ref_seq_cds, max_seq_len, tool_pkg, temperature, stability_type)
    return (results['Optimized ORF']), float(results['CAI Optimized ORF']), float(results['CAI Wild Type']), float(results['Stability of Optimized ORF']), float(results['Stability of Wild Type']), float(results['GC Content of Optimized ORF']), float(results['GC Content of Wild Type'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on the ProtBert model')
    parser.add_argument('--mask', type=bool, default=True, help='Masking for the model')
    parser.add_argument('--stability_type', type=str, default='mfe', help='Stability type (deg/mfe)')
    parser.add_argument('--tool_pkg', type=str, default='vienna', help='Tool package for stability prediction')
    parser.add_argument('--temperature', type=int, default=37, help='Temperature for stability prediction')
    parser.add_argument('--host_organism', type=str, default='human', help='Host organism for the model')
    parser.add_argument('--protein_seq', type=str, default='sars_cov2', help='Protein sequence to test')
    parser.add_argument('--model_type', type=str, default='human-long', help='Model type to use')

    args = parser.parse_args()
    run_inference(args.mask, args.stability_type, args.tool_pkg, args.temperature, args.host_organism, args.protein_seq, args.model_type)