"""
It contains all the constants used in the project

"""
NUCLEOTIDE_BASE = ['a', 't', 'c', 'g']

AMINO_ACID_DICT = {
    'A' : 1,
    'C' : 2,
    'D' : 3,
    'E' : 4,
    'F' : 5,
    'G' : 6,
    'H' : 7,
    'I' : 8,
    'K' : 9,
    'L' : 10,
    'M' : 11,
    'N' : 12,
    'P' : 13,
    'Q' : 14,
    'R' : 15,
    'S' : 16,
    'T' : 17,
    'V' : 18,
    'W' : 19,
    'Y' : 20,
}


CODON_DICT = {
    'gct' : 0,
    'gcc' : 1,
    'gca' : 2,
    'gcg' : 3,
    'cgt' : 4,
    'cgc' : 5,
    'cga' : 6,
    'cgg' : 7,
    'aga' : 8,
    'agg' : 9,
    'aat' : 10,
    'aac' : 11,
    'gat' : 12,
    'gac' : 13,
    'tgt' : 14,
    'tgc' : 15,
    'gaa' : 16,
    'gag' : 17,
    'caa' : 18,
    'cag' : 19,
    'ggt' : 20,
    'ggc' : 21,
    'gga' : 22,
    'ggg' : 23,
    'cat' : 24,
    'cac' : 25,

    'att' : 26,
    'atc' : 27,
    'ata' : 28,
    'ctt' : 29,
    'ctc' : 30,
    'cta' : 31,
    'ctg' : 32,
    'tta' : 33,
    'ttg' : 34,

    'aaa' : 35,
    'aag' : 36,
    'atg' : 37,
    'ttt' : 38,
    'ttc' : 39,
    'cct' : 40,
    'ccc' : 41,
    'cca' : 42,
    'ccg' : 43,
    'tct' : 44,
    'tcc' : 45,
    'tca' : 46,
    'tcg' : 47,
    'agt' : 48,
    'agc' : 49,
    'act' : 50,
    'acc' : 51,
    'aca' : 52,
    'acg' : 53,
    'tgg' : 54,
    'tat' : 55,
    'tac' : 56,
    'gtt' : 57,
    'gtc' : 58,
    'gta' : 59,
    'gtg' : 60,
    }

CODON_DICT_TMP = {
    'gct' : 1,
    'gcc' : 2,
    'gca' : 3,
    'gcg' : 4,
    'cgt' : 5,
    'cgc' : 6,
    'cga' : 7,
    'cgg' : 8,
    'aga' : 9,
    'agg' : 10,
    'aat' : 11,
    'aac' : 12,
    'gat' : 13,
    'gac' : 14,
    'tgt' : 15,
    'tgc' : 16,
    'gaa' : 17,
    'gag' : 18,
    'caa' : 19,
    'cag' : 20,
    'ggt' : 21,
    'ggc' : 22,
    'gga' : 23,
    'ggg' : 24,
    'cat' : 25,
    'cac' : 26,

    'att' : 27,
    'atc' : 28,
    'ata' : 29,
    'ctt' : 30,
    'ctc' : 31,
    'cta' : 32,
    'ctg' : 33,
    'tta' : 34,
    'ttg' : 35,

    'aaa' : 36,
    'aag' : 37,
    'atg' : 38,
    'ttt' : 39,
    'ttc' : 40,
    'cct' : 41,
    'ccc' : 42,
    'cca' : 43,
    'ccg' : 44,
    'tct' : 45,
    'tcc' : 46,
    'tca' : 47,
    'tcg' : 48,
    'agt' : 49,
    'agc' : 50,
    'act' : 51,
    'acc' : 52,
    'aca' : 53,
    'acg' : 54,
    'tgg' : 55,
    'tat' : 56,
    'tac' : 57,
    'gtt' : 58,
    'gtc' : 59,
    'gta' : 60,
    'gtg' : 61,
    }

AMINO_ACID = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
CODONS = ['GCT', 'GCC', 'GCA', 'GCG', 'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'AAT', 'AAC', 'GAT', 'GAC', 'TGT', 'TGC',
          'GAA', 'GAG', 'CAA', 'CAG', 'GGT', 'GGC', 'GGA', 'GGG', 'CAT', 'CAC', 'ATT', 'ATC', 'ATA', 'CTT', 'CTC', 'CTA',
          'CTG', 'TTA', 'TTG', 'AAA', 'AAG', 'ATG', 'TTT', 'TTC', 'CCT', 'CCC', 'CCA', 'CCG', 'TCT', 'TCC', 'TCA', 'TCG',
          'AGT', 'AGC', 'ACT', 'ACC', 'ACA', 'ACG', 'TGG', 'TAT', 'TAC', 'GTT', 'GTC', 'GTA', 'GTG', 'TAA', 'TAG', 'TGA' ]

SYNONYMOUS_CODONS = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],       # Alanine
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    'N': ['AAT', 'AAC'],                     # Asparagine
    'D': ['GAT', 'GAC'],                     # Aspartic Acid
    'C': ['TGT', 'TGC'],                     # Cysteine
    'E': ['GAA', 'GAG'],                     # Glutamic Acid
    'Q': ['CAA', 'CAG'],                     # Glutamine
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],       # Glycine
    'H': ['CAT', 'CAC'],                     # Histidine
    'I': ['ATT', 'ATC', 'ATA'],              # Isoleucine
    'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],  # Leucine
    'K': ['AAA', 'AAG'],                     # Lysine
    'M': ['ATG'],                            # Methionine
    'F': ['TTT', 'TTC'],                     # Phenylalanine
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],       # Proline
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],  # Serine
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],       # Threonine
    'W': ['TGG'],                            # Tryptophan
    'Y': ['TAT', 'TAC'],                     # Tyrosine
    'V': ['GTT', 'GTC', 'GTA', 'GTG']       # Valine
    # '-': ['TAA', 'TAG', 'TGA']               # Stop codons
}

# CODON BOXES Dict has changed now make necessary adjustment

codon_to_aa_cb = {
    'ATG': ('M', 'j'), 
    'GCT': ('A', 'g'), 
    'TCG': ('S', 'g'), 
    'GTG': ('V', 'k'), 
    'TTC': ('F', 'b'), 
    'TGT': ('C', 'd'), 
    'GGC': ('G', 's'), 
    'GAG': ('E', 'q'), 
    'GAC': ('D', 'r'), 
    'CTG': ('L', 'g'), 
    'AGC': ('S', 'r'), 
    'CGC': ('R', 'n'), 
    'GTA': ('V', 'j'), 
    'TGG': ('W', 'k'), 
    'AAC': ('N', 'o'), 
    'TTG': ('L', 'd'), 
    'ACC': ('T', 'm'), 
    'TCC': ('S', 'f'), 
    'ACG': ('T', 'r'), 
    'CGG': ('R', 's'), 
    'AGA': ('R', 'u'), 
    'TAC': ('Y', 'h'), 
    'AGG': ('R', 'q'), 
    'TGC': ('C', 'g'), 
    'CAC': ('H', 'm'), 
    'CAG': ('Q', 'r'), 
    'GGG': ('G', 'p'), 
    'ACA': ('T', 'o'), 
    'GAA': ('E', 'u'), 
    'CTC': ('L', 'f'), 
    'GAT': ('D', 'j'), 
    'GTT': ('V', 'd'), 
    'CCC': ('P', 'l'), 
    'AAA': ('K', 't'), 
    'CTT': ('L', 'b'), 'CTA': ('L', 'h'), 
    'GCA': ('A', 'r'), 'TCA': ('S', 'h'), 
    'CAT': ('H', 'h'), 'AAG': ('K', 'u'), 
    'TAT': ('Y', 'c'), 'TTT': ('F', 'a'), 
    'GGA': ('G', 'q'), 'ATA': ('I', 'i'), 
    'GGT': ('G', 'k'), 'TTA': ('L', 'c'), 
    'CGA': ('R', 'r'), 'CCA': ('P', 'm'), 
    'ATT': ('I', 'c'), 'AAT': ('N', 'i'), 
    'GCC': ('A', 'n'), 'CAA': ('Q', 'o'), 
    'ATC': ('I', 'h'), 'GCG': ('A', 's'), 
    'AGT': ('S', 'j'), 'ACT': ('T', 'h'), 
    'CCG': ('P', 'n'), 'GTC': ('V', 'g'), 
    'CCT': ('P', 'f'), 'CGT': ('R', 'g'), 
    'TCT': ('S', 'b')}

CODON_BOX_DICT = {
    'j': 1, 
    'g': 2, 
    'k': 3, 
    'b': 4, 
    'd': 5, 
    's': 6, 
    'q': 7, 
    'r': 8, 
    'n': 9, 
    'o': 10, 
    'm': 11, 
    'f': 12, 
    'u': 13, 
    'h': 14, 
    'p': 15, 
    'l': 16, 
    't': 17, 
    'c': 18, 
    'a': 19, 
    'i': 20
}
   

CODON_BOX_CODON_DICT = {
    'j': ['AGT', 'ATG', 'GAT', 'GTA'], 
    'g': ['CTG', 'TCG', 'TGC', 'CGT', 'GTC', 'GCT'], 
    'k': ['GGT', 'TGG', 'GTG'], 
    'b': ['CTT', 'TCT', 'TTC'], 
    'd': ['GTT', 'TTG', 'TGT'], 
    's': ['GCG', 'GGC', 'CGG'], 
    'q': ['GAG', 'GGA', 'AGG'], 
    'r': ['GCA', 'CGA', 'AGC', 'GAC', 'ACG', 'CAG'], 
    'n': ['GCC', 'CGC', 'CCG'], 
    'o': ['CAA', 'ACA', 'AAC'], 
    'm': ['CAC', 'ACC', 'CCA'], 
    'f': ['TCC', 'CTC', 'CCT'], 
    'u': ['GAA', 'AAG', 'AGA'], 
    'h': ['CAT', 'ACT', 'TCA', 'TAC', 'ATC', 'CTA'], 
    'p': ['GGG'], 
    'l': ['CCC'], 
    't': ['AAA'], 
    'c': ['TAT', 'ATT', 'TTA'], 
    'a': ['TTT'], 
    'i': ['ATA', 'AAT']
    }



    