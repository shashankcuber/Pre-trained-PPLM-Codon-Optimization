
# Pre-trained Protein Language Model for Codon Optimization (PPLM-CO)

This is the offical repository for the paper: [Pre-trained protein language model for codon optimization](https://www.biorxiv.org/content/10.1101/2024.12.12.628267v1). 

PPLM-CO in it's current version can test different pre-trained models for predicitin optimized open reading frame (ORF) sequences for input protein sequences for host organisms: Human, E.coli and Chinese Hamster Ovary (cho) cells.
**Update:** Custom Protein Sequence can now be used within the gradio app.

## Dependencies and Installation
1. Clone Repo
```
git clone https://github.com/shashankcuber/Pre-trained-PPLM-Codon-Optimization.git
```
2. Install dependent packages
```
conda create -n PPLMCO python=3.9.6 -y
conda activate PPLMCO
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
3. Download Models 
Adasel-ProtBert models are available for Human, E.coli and Chinese Hamster species.
Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1_KEn-HY4KHhrBTsHuqBV30KEXMON7TLP?usp=sharing).
Save them under the folder saved_best_model.

4. **Test Sets and Reference CAI Sets**

   Download them from [Google Drive](https://drive.google.com/drive/folders/1_KEn-HY4KHhrBTsHuqBV30KEXMON7TLP?usp=sharing).

   - Unzip the archive `Raw_Data_hg_ecoli_ch.zip`.
   - Download the following reference files:
     - `ch-mfe-2500.csv`
     - `ecoli-mfe-2500.csv`
     - `hg19_mfe_good_seq.csv`

   > **Note:** VZV and SARS-CoV-2 benchmark sequences are already included in `data_preprocessing_protbert.py`.

## Quick Inference 
```
python3 gradio_app.py
```
Open => [PPLMCO Interface](http://0.0.0.0:7860)
Server will start and in the terminal you can access the url (local or public) for opening it. 

A sample interface of the tool:
![](./assets/Inteface-PPLM-CO-Tool.png)

## Description for end user inputs
1.

|  Model Type  | Training on Species |    Dataset Filtering   |
|:------------:|:-------------------:|:----------------------:|
|     human    |        Human        | MFE and Protein Length |
|     ecoli    |        E.coli       |     Protein Length     |
|      cho     |    Chinse-Hamster   |     Protein Length     |


2. **Custom Protein Sequence, Species Specific Test Dataset or Vaccine**:
- Choose **human**, **ecoli**, or **cho** for species-specific test sets.  
- Choose **sars_cov2** (COVID-19 Spike Protein ORF) or **vzv** (Shingles vaccine ORF) for vaccine benchmarks.  
- For **custom sequences**, select `custom` from the dropdown and paste your protein sequence.  

> For **SARS-CoV-2**, the Wild-Type CAI, MFE, and GC values refer to the **Moderna (mRNA-1273) benchmark ORF**.

3. #### Host Organism
Select **human**, **ecoli**, or **cho** to evaluate the optimized ORF CAI index.

## Citation
```
@article{pathak2024pre,
  title={Pre-trained protein language model for codon optimization},
  author={Pathak, Shashank and Lin, Guohui},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```