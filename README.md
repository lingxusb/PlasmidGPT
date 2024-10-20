# PlasmidGPT: a generative framework for plasmid design and annotation
![github](https://github.com/user-attachments/assets/fc75bf4f-972c-4e3e-913e-499f01ab41ba)

We introduce PlasmidGPT, a generative language model pretrained on 153k engineered plasmid sequences from Addgene (https://www.addgene.org/). PlasmidGPT generates de novo sequences that share similar characteristics with engineered plasmids but show low sequence identity to the training data. We demonstrate its ability to generate plasmids in a controlled manner based on the input sequence or specific design constraint. Moreover, our model learns informative embeddings of both engineered and natural plasmids, allowing for efficient prediction of a wide range of sequence-related attributes.

### Installation
Python package dependencies:
- torch 2.0.1
- transformers 4.37.2
- pandas 2.2.0
- seaborn 0.13.2

We recommend using [Conda](https://docs.conda.io/en/latest/index.html) to install our packages. For convenience, we have provided a conda environment file with package versions that are compatiable with the current version of the program. The conda environment can be setup with the following comments:

1. Clone this repository:
   ```bash
     git clone https://github.com/lingxusb/PlasmidGPT.git
     cd PlasmidGPT
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f env.yml
   conda activate PlasmidGPT
   ```

### Trained model
The trained model and tokenizer is availale at [huggingface](https://huggingface.co/lingxusb/PlasmidGPT/tree/main). 
- ```pretrained_model.pt```, pretrained PlasmidGPT model, can be accessed [here](https://huggingface.co/lingxusb/PlasmidGPT/blob/main/pretrained_model.pt)
- ```addgene_trained_dna_tokenizer.json```, trained BPE tokenizer on Addgene plasmid sequences, can be accessed [here](https://huggingface.co/lingxusb/PlasmidGPT/blob/main/addgene_trained_dna_tokenizer.json)

### Sequence generation
```python
import torch

# load the model
device = 'cpu' # use 'cuda' for GPU

model = torch.load(pt_file_path).to(device)
model.eval()

# start sequence
input_ids = tokenizer.encode(start_sequence, return_tensors='pt').to(device)

# model generation
outputs = model.generate(
    input_ids,
    max_length=300,
    num_return_sequences=1,
    temperature=1.0,
    do_sample=True,
    generation_config=GenerationConfig.from_model_config(model.config)
)

# transform tokens back to DNA ucleotide sequence:
generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Please check our jupyter notebook [PlasmidGPT_generate.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebooks/PlasmidGPT_generate.ipynb). 

Or, you can easily use our [Colab Notebook](https://colab.research.google.com/drive/1xWbekcTpcGMSiQE6LkRnqSTjswDkKAoc?usp=sharing) in the browser. Please make sure to connect to a GPU instance (e.g., T4 GPU). The notebook will automatically download the pretrained model and tokenizer. The plasmid sequence can be generated based on the user's specified start sequence and downloaded in the ```.fasta``` file format.


### Model embeddings
```python
# calculation of model embeddings
model.config.output_hidden_states = True

# Inference to obtain hidden states
with torch.no_grad():
    outputs = model(input_ids)
    hidden_states = outputs.hidden_states[-1].cpu().numpy()
    hidden_states_mean = np.mean(hidden_states, axis=1).reshape(-1)    
    embedding.append(hidden_states_mean)
```
### Sequence annotation
For prediction of attributes, please check our models in the ```prediction_models``` folder

We have provided the jupyter notebook [PlasmidGPT_predict.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebooks/PlasmidGPT_predict.ipynb) for the prediction of lab of origin.

The [Colab Notebook](https://colab.research.google.com/drive/1vo27RBnScf_cOISBdd13YN_hr5-ZVNHx?usp=sharing) can be easily used in the browser to predict the lab of origin, species, and vector type for the input sequence. The notebook will automatically download all related models and make predictions based on the user's input plasmid sequence. Please use the drop-down list to select the feature to predict, and the top 10 predictions will be displayed.


### Reference
- [PlasmidGPT: a generative framework for plasmid design and annotation](https://www.biorxiv.org/content/10.1101/2024.09.30.615762v1)
