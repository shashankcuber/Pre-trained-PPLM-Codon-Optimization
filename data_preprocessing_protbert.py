import json
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS
# import sys

random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class CodonDataset(Dataset):
    def __init__(self, aa_sequences, cds_sequences):
        self.aa_seqs = [torch.tensor(seq) for seq in aa_sequences]
        self.cds_seqs = [torch.tensor(seq) for seq in cds_sequences]
    
    def __len__(self):
        return len(self.aa_seqs)

    def __getitem__(self, idx):
        aa_seq = self.aa_seqs[idx]
        cds_seq = self.cds_seqs[idx]
        return aa_seq, cds_seq


class AminoAcidCodonDataset(Dataset):
    def __init__(self, aa_sequences, codon_sequences, max_len):
        self.aa_sequences = aa_sequences
        self.codon_sequences = codon_sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.aa_sequences)

    def tokenize_aa_sequence(self, aa_seq):
        return [AMINO_ACID_DICT[aa] for aa in aa_seq]  # Map amino acids to integers

    def tokenize_codon_sequence(self, codon_seq):
        codons = [codon_seq[i:i+3] for i in range(0, len(codon_seq), 3)]
        return [CODON_DICT.get(codon, 0) for codon in codons]  # Map codons to integers

    def create_attention_mask(self, seq_length):
        return [1] * seq_length + [0] * (self.max_len - seq_length)

    def __getitem__(self, idx):
        aa_seq = self.aa_sequences[idx]
        codon_seq = self.codon_sequences[idx]

        aa_tokens = self.tokenize_aa_sequence(aa_seq)
        codon_tokens = self.tokenize_codon_sequence(codon_seq)
        # print("lllll", len(codon_tokens))
        # Create attention mask
        attention_mask = self.create_attention_mask(len(aa_tokens))

        # Padding
        aa_tokens += [0] * (self.max_len - len(aa_tokens))
        codon_tokens += [-100] * (self.max_len - len(codon_tokens))

        return {
            'input_ids': torch.tensor(aa_tokens[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(codon_tokens[:self.max_len], dtype=torch.long)
        }

def read_data_from_file(data_file_path):
    
    with open(data_file_path) as f:
        cds_aa_dict = json.load(f)
    return cds_aa_dict

def collate_fn(batch):
    aa_seqs, cds_seqs = zip(*batch)
    aa_seqs_padded = pad_sequence(aa_seqs, batch_first=True, padding_value=0)
    cds_seqs_padded = pad_sequence(cds_seqs, batch_first=True, padding_value=-100)
    
    return aa_seqs_padded, cds_seqs_padded

def get_tokenized_padded_data(data_preprocess_obj):
    
    # print("\n")
    # print("################### PREPROCESSING START ####################### \n")
    #data preprocessing
    aa_seq_tokenized, cds_seq_tokenized, aa_token_dict, cds_token_dict = data_preprocess_obj.preprocess()
    # print("\n")
    # print("Dimension of Tokenized and processesed AA: ", (aa_seq_tokenized.shape))
    # print("\n")
    # print("Dimension of Tokenized and processesed CDS: ", (cds_seq_tokenized.shape))
    # print("\n")
    
    # print("################### PREPROCESSING DONE ######################## \n")
    return aa_seq_tokenized, cds_seq_tokenized, aa_token_dict, cds_token_dict

def get_train_test_split(aa_seq_tokenized, cds_seq_tokenized, split_ratio):
   
    # print("################### STARTING TRAIN TEST SPLIT ######################### \n")
    aa_train, aa_test, cds_train, cds_test = train_test_split(
        aa_seq_tokenized, cds_seq_tokenized, test_size=split_ratio, random_state=42
        )
    print("Number of training + validation samples are ", len(aa_train))
    print("Number of test samples are ", len(aa_test))
    # print("\n")
    return aa_train, aa_test, cds_train, cds_test

def get_train_validation_split(aa_train, cds_train, split_ratio):
    
    # print("################### STARTING TRAIN VALIDATION SPLIT ################### \n")
    aa_train, aa_val, cds_train, cds_val = train_test_split(
        aa_train, cds_train, test_size=split_ratio, random_state=42 
    )
    print("Number of training samples are ", len(aa_train))
    print("Number of validation samples are ", len(aa_val))
    # print("\n")
    return aa_train, aa_val, cds_train, cds_val

def get_train_val_test_dataset(aa_train, aa_val, aa_test, cds_train, cds_val, cds_test):
    
    # print("################### STARTING DATASET GENERATION ####################### \n")
    # train_dataset = CodonDataset(aa_train, cds_train)
    # val_dataset = CodonDataset(aa_val, cds_val)
    # test_dataset = CodonDataset(aa_test, cds_test)
    train_dataset = TensorDataset(aa_train, cds_train)
    val_dataset = TensorDataset(aa_val, cds_val)
    test_dataset = TensorDataset(aa_test, cds_test)

    # print("################### DATASET GENERATION DONE ########################### \n")
    return train_dataset, val_dataset, test_dataset

def get_train_val_test_dataloaders(train_dataset, val_dataset, test_dataset):
    
    # print("################### STARTING DATALOADERS GENERATION ################### \n")
    train_loader = DataLoader(
        train_dataset, batch_size = 64, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size = 64, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size = 64, shuffle=False
    )
    # print("################### DATALOADERS GENERATION DONE ####################### \n")
    return train_loader, val_loader, test_loader

def start_preprocessing_probert(data_file_path, protein_seq ='', host_organism='human'):
  
    cds_aa_dict = read_data_from_file(data_file_path)
    # making separate list of 18100 AA and CDS sequences
    cds_list = list(cds_aa_dict.keys())
    aa_list = list(cds_aa_dict.values())
    print(f"Host Organism: {host_organism}")
    # 5440- ecoli and 30k - ch and 18100- hg19 
    if host_organism == 'human':
        ref_cds_list = random.sample(cds_list, 18100)
    elif host_organism == 'chinese_hamster':
        ref_cds_list = random.sample(cds_list, 30000)
    elif host_organism == 'ecoli':
        ref_cds_list = random.sample(cds_list, 5440)
    else:
        ref_cds_list = random.sample(cds_list, 18100)


    if protein_seq =='sars_cov2':
        # https://www.ncbi.nlm.nih.gov/nuccore/OR134578.1?report=fasta  This is where the sequence is taken from by submitting OR134578.1 in ORF Finder
        amino_seq_moderna = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAISGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQQVFAQVKQIYKTPPIKYFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSNFGAISSVLNDILSRLDPPEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
        sequence_moderna = "ATGTTCGTGTTCCTGGTGCTGCTGCCCCTGGTGAGCAGCCAGTGCGTGAACCTGACCACCCGGACCCAGCTGCCACCAGCCTACACCAACAGCTTCACCCGGGGCGTCTACTACCCCGACAAGGTGTTCCGGAGCAGCGTCCTGCACAGCACCCAGGACCTGTTCCTGCCCTTCTTCAGCAACGTGACCTGGTTCCACGCCATCAGCGGCACCAACGGCACCAAGCGGTTCGACAACCCCGTGCTGCCCTTCAACGACGGCGTGTACTTCGCCAGCACCGAGAAGAGCAACATCATCCGGGGCTGGATCTTCGGCACCACCCTGGACAGCAAGACCCAGAGCCTGCTGATCGTGAATAACGCCACCAACGTGGTGATCAAGGTGTGCGAGTTCCAGTTCTGCAACGACCCCTTCCTGGGCGTGTACTACCACAAGAACAACAAGAGCTGGATGGAGAGCGAGTTCCGGGTGTACAGCAGCGCCAACAACTGCACCTTCGAGTACGTGAGCCAGCCCTTCCTGATGGACCTGGAGGGCAAGCAGGGCAACTTCAAGAACCTGCGGGAGTTCGTGTTCAAGAACATCGACGGCTACTTCAAGATCTACAGCAAGCACACCCCAATCAACCTGGTGCGGGATCTGCCCCAGGGCTTCTCAGCCCTGGAGCCCCTGGTGGACCTGCCCATCGGCATCAACATCACCCGGTTCCAGACCCTGCTGGCCCTGCACCGGAGCTACCTGACCCCAGGCGACAGCAGCAGCGGGTGGACAGCAGGCGCGGCTGCTTACTACGTGGGCTACCTGCAGCCCCGGACCTTCCTGCTGAAGTACAACGAGAACGGCACCATCACCGACGCCGTGGACTGCGCCCTGGACCCTCTGAGCGAGACCAAGTGCACCCTGAAGAGCTTCACCGTGGAGAAGGGCATCTACCAGACCAGCAACTTCCGGGTGCAGCCCACCGAGAGCATCGTGCGGTTCCCCAACATCACCAACCTGTGCCCCTTCGACGAGGTGTTCAACGCCACCCGGTTCGCCAGCGTGTACGCCTGGAACCGGAAGCGGATCAGCAACTGCGTGGCCGACTACAGCGTGCTGTACAACTTCGCCCCATTCTTCGCCTTCAAGTGCTACGGCGTGAGCCCCACCAAGCTGAACGACCTGTGCTTCACCAACGTGTACGCCGACAGCTTCGTGATCCGTGGCAACGAGGTGAGCCAGATCGCACCCGGCCAGACAGGCAACATCGCCGACTACAACTACAAGCTGCCCGACGACTTCACCGGCTGCGTGATCGCCTGGAACAGCAACAAGCTCGACAGCAAGGTGGGCGGCAACTACAACTACCGGTACCGGCTGTTCCGGAAGAGCAACCTGAAGCCCTTCGAGCGGGACATCAGCACCGAGATCTACCAAGCCGGCAACAAGCCTTGCAACGGCGTGGCCGGCGTGAACTGCTACTTCCCTCTGCAGAGCTACGGCTTCCGGCCCACCTACGGCGTGGGCCACCAGCCCTACCGGGTGGTGGTGCTGAGCTTCGAGCTGCTGCACGCCCCAGCCACCGTGTGTGGCCCCAAGAAGAGCACCAACCTGGTGAAGAACAAGTGCGTGAACTTCAACTTCAACGGCCTTACCGGCACCGGCGTGCTGACCGAGAGCAACAAGAAATTCCTGCCCTTTCAGCAGTTCGGCCGGGACATCGCCGACACCACCGACGCTGTGCGGGATCCCCAGACCCTGGAGATCCTGGACATCACCCCTTGCAGCTTCGGCGGCGTGAGCGTGATCACCCCAGGCACCAACACCAGCAACCAGGTGGCCGTGCTGTACCAGGGTGTGAACTGCACCGAGGTGCCCGTGGCCATCCACGCCGACCAGCTGACACCCACCTGGCGGGTCTACAGCACCGGCAGCAACGTGTTCCAGACCCGGGCCGGTTGCCTGATCGGCGCCGAGCACGTGAACAACAGCTACGAGTGCGACATCCCCATCGGCGCCGGCATCTGTGCCAGCTACCAGACCCAGACCAAGTCACACCGGAGGGCAAGGAGCGTGGCCAGCCAGAGCATCATCGCCTACACCATGAGCCTGGGCGCCGAGAACAGCGTGGCCTACAGCAACAACAGCATCGCCATCCCCACCAACTTCACCATCAGCGTGACCACCGAGATTCTGCCCGTGAGCATGACCAAGACCAGCGTGGACTGCACCATGTACATCTGCGGCGACAGCACCGAGTGCAGCAACCTGCTGCTGCAGTACGGCAGCTTCTGCACCCAGCTGAACCGGGCCCTGACCGGCATCGCCGTGGAGCAGGACAAGAACACCCAGCAGGTGTTCGCCCAGGTGAAGCAGATCTACAAGACCCCTCCCATCAAGTACTTCGGCGGCTTCAACTTCAGCCAGATCCTGCCCGACCCCAGCAAGCCCAGCAAGCGGAGCTTCATCGAGGACCTGCTGTTCAACAAGGTGACCCTAGCCGACGCCGGCTTCATCAAGCAGTACGGCGACTGCCTCGGCGACATAGCCGCCCGGGACCTGATCTGCGCCCAGAAGTTCAACGGCCTGACCGTGCTGCCTCCCCTGCTGACCGACGAGATGATCGCCCAGTACACCAGCGCCCTGTTAGCCGGAACCATCACCAGCGGCTGGACTTTCGGCGCTGGAGCCGCTCTGCAGATCCCCTTCGCCATGCAGATGGCCTACCGGTTCAACGGCATCGGCGTGACCCAGAACGTGCTGTACGAGAACCAGAAGCTGATCGCCAACCAGTTCAACAGCGCCATCGGCAAGATCCAGGACAGCCTGAGCAGCACCGCTAGCGCCCTGGGCAAGCTGCAGGACGTGGTGAACCACAACGCCCAGGCCCTGAACACCCTGGTGAAGCAGCTGAGCAGCAACTTCGGCGCCATCAGCAGCGTGCTGAACGACATCCTGAGCCGGCTGGACCCTCCCGAGGCCGAGGTGCAGATCGACCGGCTGATCACTGGCCGGCTGCAGAGCCTGCAGACCTACGTGACCCAGCAGCTGATCCGGGCCGCCGAGATTCGGGCCAGCGCCAACCTGGCCGCCACCAAGATGAGCGAGTGCGTGCTGGGCCAGAGCAAGCGGGTGGACTTCTGCGGCAAGGGCTACCACCTGATGAGCTTTCCCCAGAGCGCACCCCACGGAGTGGTGTTCCTGCACGTGACCTACGTGCCCGCCCAGGAGAAGAACTTCACCACCGCCCCAGCCATCTGCCACGACGGCAAGGCCCACTTTCCCCGGGAGGGCGTGTTCGTGAGCAACGGCACCCACTGGTTCGTGACCCAGCGGAACTTCTACGAGCCCCAGATCATCACCACCGACAACACCTTCGTGAGCGGCAACTGCGACGTGGTGATCGGCATCGTGAACAACACCGTGTACGATCCCCTGCAGCCCGAGCTGGACAGCTTCAAGGAGGAGCTGGACAAGTACTTCAAGAATCACACCAGCCCCGACGTGGACCTGGGCGACATCAGCGGCATCAACGCCAGCGTGGTGAACATCCAGAAGGAGATCGATCGGCTGAACGAGGTGGCCAAGAATCTGAACGAGAGCCTGATCGACCTGCAAGAACTGGGGAAGTACGAGCAGTACATCAAGTGGCCCTGGTACATCTGGCTGGGCTTTATCGCCGGACTGATTGCCATCGTGATGGTCACAATCATGCTGTGTTGCATGACCAGCTGCTGTAGCTGCCTGAAGGGCTGTTGTAGCTGTGGCAGCTGCTGCAAGTTCGACGAGGACGATTCTGAGCCCGTGCTGAAGGGCGTGAAACTGCACTACACA"
        # print(len(amino_seq_moderna))
        # print(len(sequence_moderna))
        # print("MODERNA")
    
        sequence_moderna = sequence_moderna.lower()
        max_len = len(sequence_moderna)
        cds_list = [sequence_moderna, sequence_moderna, sequence_moderna, sequence_moderna, sequence_moderna]
        aa_list = [amino_seq_moderna, amino_seq_moderna, amino_seq_moderna, amino_seq_moderna, amino_seq_moderna]
   
    elif protein_seq == 'vzv':
        """
        VZV vaccine results
        """
        print("VZV vaccine")
        # Taken from supplementary material of LinearDesign
        amino_seq_vzv = "MGTVNKPVVGVLMGFGIITGTLRITNPVRASVLRYDDFHIDEDKLDTNSVYEPYYHSDHAESSWVNRGESSRKAYDHNSPYIWPRNDYDGFLENAHEHHGVYNQGRGIDSGERLMQPTQMSAQEDLGDDTGIHVIPTLNGDDRHKIVNVDQRQYGDVFKGDLNPKPQGQRLIEVSVEENHPFTLRAPIQRIYGVRYTETWSFLPSLTCTGDAAPAIQHICLKHTTCFQDVVVDVDCAENTKEDQLAEISYRFQGKKEADQPWIVVNTSTLFDELELDPPEIEPGVLKVLRTEKQYLGVYIWNMRGSDGTSTYATFLVTWKGDEKTRNPTPAVTPQPRGAEFHMWNYHSHVFSVGDTFSLAMHLQYKIHEAPFDLLLEWLYVPIDPTCQPMRLYSTCLYHPNAPQCLSHMNSGCTFTSPHLAQRVASTVYQNCEHADNYTAYCLGISHMEPSFGLILHDGGTTLKFVDTPESLSGLYVFVVYFNGHVEAVAYTVVSTVDHFVNAIEERGFPPTAGQPPATTKPKEITPVNPGTSPLLRYAAWTGGLAAVVLLCLVIFLICTAKRMRVKAYRVDKSPYNQSMYYAGLPVDDFEDSESTDTEEEFGNAIGGSHGGSSYTVYIDKTR"
        sequence_vzv_wt = "atggggacagttaataaacctgtggtgggggtattgatggggttcggaattatcacgggaacgttgcgtataacgaatccggtcagagcatccgtcttgcgatacgatgattttcacatcgatgaagacaaactggatacaaactccgtatatgagccttactaccattcagatcatgcggagtcttcatgggtaaatcggggagagtcttcgcgaaaagcgtacgatcataactcaccttatatatggccacgtaatgattatgatggatttttagagaacgcacacgaacaccatggggtgtataatcagggccgtggtatcgatagcggggaacggttaatgcaacccacacaaatgtctgcacaggaggatcttggggacgatacgggcatccacgttatccctacgttaaacggcgatgacagacataaaattgtaaatgtggaccaacgtcaatacggtgacgtgtttaaaggagatcttaatccaaaaccccaaggccaaagactcattgaggtgtcagtggaagaaaatcacccgtttactttacgcgcaccgattcagcggatttatggagtccggtacaccgagacttggagctttttgccgtcattaacctgtacgggagacgcagcgcccgccatccagcatatatgtttaaaacatacaacatgctttcaagacgtggtggtggatgtggattgcgcggaaaatactaaagaggatcagttggccgaaatcagttaccgttttcaaggtaagaaggaagcggaccaaccgtggattgttgtaaacacgagcacactgtttgatgaactcgaattagacccccccgagattgaaccgggtgtcttgaaagtacttcggacagaaaaacaatacttgggtgtgtacatttggaacatgcgcggctccgatggtacgtctacctacgccacgtttttggtcacctggaaaggggatgaaaaaacaagaaaccctacgcccgcagtaactcctcaaccaagaggggctgagtttcatatgtggaattaccactcgcatgtattttcagttggtgatacgtttagcttggcaatgcatcttcagtataagatacatgaagcgccatttgatttgctgttagagtggttgtatgtccccatcgatcctacatgtcaaccaatgcggttatattctacgtgtttgtatcatcccaacgcaccccaatgcctctctcatatgaattccggttgtacatttacctcgccacatttagcccagcgtgttgcaagcacagtgtatcaaaattgtgaacatgcagataactacaccgcatattgtctgggaatatctcatatggagcctagctttggtctaatcttacacgacgggggcaccacgttaaagtttgtagatacacccgagagtttgtcgggattatacgtttttgtggtgtattttaacgggcatgttgaagccgtagcatacactgttgtatccacagtagatcattttgtaaacgcaattgaagagcgtggatttccgccaacggccggtcagccaccggcgactactaaacccaaggaaattacccccgtaaaccccggaacgtcaccacttctacgatatgccgcatggaccggagggcttgcagcagtagtacttttatgtctcgtaatatttttaatctgtacggctaaacgaatgagggttaaagcctatagggtagacaagtccccgtataaccaaagcatgtattacgctggccttccagtggacgatttcgaggactcggaatctacggatacggaagaagagtttggtaacgcgattggagggagtcacgggggttcgagttacacggtgtatatagataagacccgg"
        max_len = len(sequence_vzv_wt)
        cds_list = [sequence_vzv_wt, sequence_vzv_wt, sequence_vzv_wt, sequence_vzv_wt, sequence_vzv_wt]
        aa_list = [amino_seq_vzv, amino_seq_vzv, amino_seq_vzv, amino_seq_vzv, amino_seq_vzv]

    elif protein_seq == 'cho':
        max_len = 500
        print("Chinese Hamster test set")
        ch_df = pd.read_csv('./ch-mfe-2500.csv')
        ch_df = ch_df[ch_df['length']<= max_len]
        cds_list = ch_df['orf_sequence'].tolist()
        aa_list = ch_df['aa_sequence'].tolist()

    elif protein_seq == 'ecoli':
        max_len = 500
        print("Ecoli test set")
        ecoli_df = pd.read_csv('./ecoli-mfe-2500.csv')
        ecoli_df = ecoli_df[ecoli_df['length']<= max_len]
        cds_list = ecoli_df['orf_sequence'].tolist()
        aa_list = ecoli_df['aa_sequence'].tolist()
    # elif protein_seq == 'human_short':
    #     max_len = 500
    #     print("Human test set")
    #     hg19_df = pd.read_csv('./hg19_mfe_good_seq.csv')
    #     hg19_df = hg19_df[hg19_df['length']<= max_len]
    #     cds_list = hg19_df['orf_sequence'].tolist()
    #     aa_list = hg19_df['aa_sequence'].tolist()
    #     cds_list = [x[:-3] for x in cds_list]
    elif protein_seq == 'human':
        max_len = 2499
        print("Human long protein test set")
        # hg19_df = pd.read_csv('./hg19-long-filtered-mfe.csv')
        # hg19_df = hg19_df[hg19_df['length']<= max_len]
        # hg19_df = hg19_df[hg19_df['mfe']<-0.30]
        # cds_list = list(hg19_df['orf_sequence'])
        # aa_list = list(hg19_df['aa_sequence'])
        # cds_list = [x[:-3] for x in cds_list]
        
        # temporarily make the test_aa_human, test_cds_human
        test_human = pd.read_csv('./test_set_human_long_protbert.csv')
        test_aa_human = test_human['aa_sequence'].tolist()
        test_cds_human = test_human['cds_sequence'].tolist()
        test_cds_human = [x[:-3] for x in test_cds_human]
    else:
        print("Custom Protein Sequence")
        amino_seq = protein_seq.upper()
        # codon_seq = ''
        # make a dummy valid codon sequence for data preprocessing and further handling of functions during inference that require codon sequence
        codon_seq = ''.join([SYNONYMOUS_CODONS[aa][0] for aa in amino_seq])  # Use the first codon for each amino acid
        max_len = len(amino_seq) * 3
        assert len(codon_seq) == max_len, "Codon sequence length does not match amino acid sequence length"
        cds_list = [codon_seq, codon_seq, codon_seq, codon_seq, codon_seq]
        aa_list = [amino_seq, amino_seq, amino_seq, amino_seq, amino_seq]
    
    print("Total sequences ", len(cds_list))
   
    cds_list = cds_list[:]
    aa_list = aa_list[:]

    
    train_val_aa, test_aa, train_val_cds, test_cds = train_test_split(aa_list, cds_list, test_size=0.2)
    # save the test aa and test cds sequences to a csv file for future reference
    # if protein_seq == 'human_long':
    #     pd.DataFrame({'aa_sequence': test_aa, 'cds_sequence': test_cds}).to_csv(f'./test_set_human_long_protbert.csv', index=False)
    #     sys.exit()
    train_aa, val_aa, train_cds, val_cds = train_test_split(train_val_aa, train_val_cds, test_size=0.2)

    train_dataset = AminoAcidCodonDataset(train_aa, train_cds, max_len)
    val_dataset = AminoAcidCodonDataset(val_aa, val_cds, max_len)
    test_dataset = AminoAcidCodonDataset(test_aa, test_cds, max_len)
    
    if protein_seq == 'human':
        test_dataset_human = AminoAcidCodonDataset(test_aa_human, test_cds_human, max_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Length of train_loader: ", len(train_loader.dataset))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Length of val_loader: ", len(val_loader.dataset))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Length of test_loader: ", len(test_loader.dataset))
    
    if protein_seq == 'human':
        test_human_loader = DataLoader(test_dataset_human, batch_size=1, shuffle=False)
        print("Length of test_human_loader: ", len(test_human_loader.dataset))
    
        return train_loader, val_loader, test_human_loader, ref_cds_list, CODON_DICT, max_len

    return train_loader, val_loader, test_loader, ref_cds_list, CODON_DICT, max_len