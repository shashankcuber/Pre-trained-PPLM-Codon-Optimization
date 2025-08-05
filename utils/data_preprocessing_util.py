import json
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, CODON_BOX_DICT

class DataPreprocessing:
    def __init__(self, aa_list, cds_list):
        self.aa_list = aa_list
        self.cds_list = cds_list
        

    # make AA sequence and CDS sequence as sequence of words
    # For eg: AA_seq = GKLMA -> encrypt -> G K L M A ,gap of n=1 for breaking into words
    # For eg: CDS_seq = atggct -> encrypt -> atg gct , gap of n=3 for breaking into words
    def encrypt(self, sequence, n):
        return ' '.join([sequence[i:i+n] for i in range(0, len(sequence), n)])

    # pad post sequences with 0's with maxm length sequence
    def pad(self, sequences, max_len = None):
        #padding the pytorch way !!
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        
        return padded_sequences

    def tokenize(self, sequences, aa=False):

        if aa:
            # print("AA_tokenizer")
            token_to_index = AMINO_ACID_DICT
        else:
            token_to_index = CODON_DICT

        # Convert sequences to their index representation
        sequences_index = [[token_to_index[token] for token in sequence.split(' ')] for sequence in sequences]
        return sequences_index, token_to_index

    # Preprocess data
    def preprocess(self):
        # # print("Before making them words original list >>>>>>>>> \n")
        # for i in range(0,2):
        #     print("AA_seq: ", len(self.aa_list[i]))
        #     print("CDS_seq: ", len(self.cds_list[i]))
        #     print("\n")
        
        aa_words = [self.encrypt(aa_seq, 1) for aa_seq in self.aa_list]
        cds_words = [self.encrypt(cds_seq, 3) for cds_seq in self.cds_list]

        # print("After making them words >>>>>>>>> \n")
        # for i in range(0,3):
        #     print("AA_seq: ", len(aa_words[i]))
        #     print("CDS_seq: ", len(cds_words[i]))
        #     print("\n")
        
        aa_seq_tokenized, aa_token_dict = self.tokenize(aa_words, aa=True)
        
        cds_seq_tokenized, cds_token_dict = self.tokenize(cds_words, aa=False)
        # print("Amino Acid Token Dictionary >>>>>>>>>> \n",aa_token_dict)
        # print("\n")
        # print("CDS token dictionary>>>>>>>>>>> \n", cds_token_dict)
        # print("\n")
        
        # print("After tokenization >>>>>>>>> \n")
        # for i in range(0,3):
        #     print("AA_seq: ", len(aa_seq_tokenized[i]))
        #     print("CDS_seq: ", len(cds_seq_tokenized[i]))
        #     print("\n")

        # padding the sequences as they can be of variable length
        # aa_tokenized_padded = self.pad([torch.tensor(seq) for seq in aa_seq_tokenized])
        # cds_tokenized_padded= self.pad([torch.tensor(seq) for seq in cds_seq_tokenized])

        # print("AFTER PADDING >>>>>>>> \n")
        # for i in range(0,3):
        #     print("AA_seq: ", len(aa_tokenized_padded[i]))
        #     print("CDS_seq: ", len(cds_tokenized_padded[i]))
        #     print("\n")
        
        # print("\n")
        # print("AA padded sample --->", aa_tokenized_padded)
        # print("\n")
        # print("CDS padded sample --->", cds_tokenized_padded)
        # aa_seq_tokenized = np.array(aa_seq_tokenized)
        # cds_seq_tokenized = np.array(cds_seq_tokenized)

        # cds_tokenized_padded = cds_tokenized_padded.reshape(*cds_tokenized_padded.shape , 1)
        return aa_seq_tokenized, cds_seq_tokenized, aa_token_dict, cds_token_dict
        # return aa_tokenized_padded, cds_tokenized_padded, aa_token_dict, cds_token_dict

if __name__ == '__main__':
    data_file_path = './aa_cds_dict.json'
    with open(data_file_path) as f:
        cds_aa_dict = json.load(f)
    obj = DataPreprocessing(list(cds_aa_dict.values())[:2], list(cds_aa_dict.keys())[:2])
    aa_seq_tokenized, cb_seq_tokenized, aa_token_dict, cds_token_dict = obj.preprocess()