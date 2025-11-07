from utils.CONSTANTS import NUCLEOTIDE_BASE, AMINO_ACID, codon_to_aa_cb


class SequenceFilterTest:
    def __init__(self, data, total_test_failed=None):
        self.df = data  
        self.sequences_failed_id = []      
        self.nucleotide_base = NUCLEOTIDE_BASE
        self.amino_acid = AMINO_ACID
        self.total_test_failed = total_test_failed

    def extract_cds_aa(self):
        cds_sequences = self.df['orf_sequence'].tolist()
        aa_sequences = self.df['aa_sequence'].tolist()
        return cds_sequences, aa_sequences

    def is_valid_aa_orf_pair(self, amino_acid_sequence, codon_sequence):
        """
        Check if the amino acid sequence and codon sequence are valid pairs.
        """
        # if len(amino_acid_sequence) * 3 != len(codon_sequence[:-3]):
        #     return False
        
        for i in range(len(amino_acid_sequence)):
            codon = codon_sequence[i * 3: (i + 1) * 3]
            amino_acid = amino_acid_sequence[i]
            valid_amino = ''
            
            if codon.upper() in codon_to_aa_cb:
                valid_amino = codon_to_aa_cb[codon.upper()][0]
            
            if valid_amino != amino_acid:
                # print(f'Codon is {codon} with aa = {amino_acid} and valid aa = {valid_amino}')
                return False
            # if codon_to_amino_acid.get(codon, '') != amino_acid_sequence[i]:
            #     return False

        return True

    def start_test(self):
        
        # extract the cds and amino acid sequence
        cds_sequences, aa_sequences = self.extract_cds_aa()
        # cds_sequences = cds_sequences[:1]
        for i in range(len(cds_sequences)):
            # reinitialize the tests for each new sequence
            # 0 means pass and 1 mean fail
            tests_failed = {
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0
            }
            cds_seq = cds_sequences[i]
            aa_seq = aa_sequences[i]

            # Test 1: ORF length is divisible by 3
            if len(cds_seq) % 3 != 0:
                tests_failed['1'] = 1
                self.total_test_failed['1'] += 1
            
            # Test 2: CDS sequence contains only A, T, C, G
            for base in cds_seq:
                if base not in self.nucleotide_base:
                    print("NUCLEOTIDE BASE INVALID>>>:" , base)
                    tests_failed['2'] = 1
                    self.total_test_failed['2'] += 1
                    break
            
            # Test 3: Start Codon is ATG and M is the first amino acid
            if cds_seq[:3] != 'atg' or aa_seq[0] != 'M':
                tests_failed['3'] = 1
                self.total_test_failed['3'] += 1
                # print(self.df['ncbi_id'][i])
            
            # Test 4: Stop Codon is TAA, TAG, TGA
            if cds_seq[-3:] not in ['taa', 'tag', 'tga']:
                tests_failed['4'] = 1
                # print(self.df['ncbi_id'][i])
                self.total_test_failed['4'] += 1
            
           
            # Test 5 is not required as we are not using the stop codon in the amino acid sequence and hence it tests valid pair of amino acid and cds sequence
            if self.is_valid_aa_orf_pair(aa_seq, cds_seq) == False:
                tests_failed['5'] = 1
                self.total_test_failed['5'] += 1
                # print("INVALID PAIR: ", self.df['ncbi_orf_id'][i])
            
            # Test 6: Amino Acid sequence contains only amino acid in the list amino_acid
            for aa in aa_seq:
                if aa not in self.amino_acid:
                    tests_failed['6'] = 1
                    self.total_test_failed['6'] += 1
                    break
            
            # Test 7: Amnio Acid Sequence length is multiple of 3 of CDS sequence length
            if len(aa_seq) * 3 != len(cds_seq[:-3]):
                tests_failed['7'] = 1
                self.total_test_failed['7'] += 1
            
            # Append the sequence id to the list if any of the test fails
            num_tests_failed = sum(tests_failed.values())
        
            if num_tests_failed > 0:
                self.sequences_failed_id.append(self.df['ncbi_orf_id'][i])

            print(f'Tests failed for the {i}th sequence : ', tests_failed)

        return self.sequences_failed_id, self.total_test_failed