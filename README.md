# PlasmidGPT: a generative framework for plasmid design and annotation
![github](https://github.com/user-attachments/assets/fc75bf4f-972c-4e3e-913e-499f01ab41ba)

We introduce PlasmidGPT, a generative language model pretrained on 153k engineered plasmid sequences from Addgene (https://www.addgene.org/). PlasmidGPT generates de novo sequences that share similar characteristics with engineered plasmids but show low sequence identity to the training data. We demonstrate its ability to generate plasmids in a controlled manner based on the input sequence or specific design constraint. Moreover, our model learns informative embeddings of both engineered and natural plasmids, allowing for efficient prediction of a wide range of sequence-related attributes.

## Table of Contents

- [Installation](#Installation)
- [Trained model](#Trained-model)
- [Sequence generation](#Sequence-generation)
- [Model embeddings](#Model-embeddings)
- [Sequence annotation](#Sequence-annotation)

## Installation
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
   ```bash
   conda env create -f env.yml
   conda activate PlasmidGPT
   ```

## Trained model
The trained model and tokenizer is availale at [huggingface](https://huggingface.co/lingxusb/PlasmidGPT/tree/main).
- ```pretrained_model.pt```, pretrained PlasmidGPT model, can be accessed [here](https://huggingface.co/lingxusb/PlasmidGPT/blob/main/pretrained_model.pt)
- ```addgene_trained_dna_tokenizer.json```, trained BPE tokenizer on Addgene plasmid sequences, can be accessed [here](https://huggingface.co/lingxusb/PlasmidGPT/blob/main/addgene_trained_dna_tokenizer.json)


## Sequence generation
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
### command line
To generate plasmid sequence using the model, please run the following command:
```Python
python generate.py --model_dir ../pretrained_model
```
The ```../pretrained_model``` folder should contain the model file and the tokenizer.

For a full list of options, please run:

```
python generate.py -h
```

Arguments description

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
| ```-m```, ```--model_dir```  | path to the directory containing the pretrained model and tokenizer, __required__  |
|  ```-s```, ```--start_sequence```| starting DNA sequence for sequence generation  |
| ```-f```, ```--fasta_file```| FASTA file containing the starting sequence  |
| ```-n```, ```--num_sequences``` | number of sequences to generate, default value: 1  |
| ```-l```, ```--max_length``` | maximum length of the tokenized generated sequence, default value: 300 |
| ```-t```, ```--temperature``` | temperature for sequence generation (controls randomness), default value: 1.0  |
| ```-o```, ```--output``` | output file name for the generated sequences, default value: generated_sequence.fasta|

The model output will be stored in the ```generated_sequence.fasta``` file. The script should automatically detect whether to use CUDA (GPU) or CPU based on availability. If you encounter a CUDA-related error when running on a CPU-only machine, the script will handle this by falling back to CPU.


### notebooks
Please also check our jupyter notebook [PlasmidGPT_generate.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebooks/PlasmidGPT_generate.ipynb).

Or, you can easily use our [Colab Notebook](https://colab.research.google.com/drive/1xWbekcTpcGMSiQE6LkRnqSTjswDkKAoc?usp=sharing) in the browser. Please make sure to connect to a GPU instance (e.g., T4 GPU). The notebook will automatically download the pretrained model and tokenizer. The plasmid sequence can be generated based on the user's specified start sequence and downloaded in the ```.fasta``` file format.


## Model embeddings
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
### command
To generate plasmid sequence embeddings, please run the following command:
```Python
python embeddings.py [-h] -m MODEL_DIR -f FASTA_FILE [-o OUTPUT_FILE]
```

Arguments description

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
| ```-m```, ```--model_dir```  | path to the directory containing the pretrained model and tokenizer, __required__  |
|  ```-f```, ```--fasta_file```| FASTA file containing DNA sequences for the embedding calculation, __required__  |
| ```-o```, ```--output_file```| output file name for saving the embeddings |

The model output will be save in the ```embeddings.txt``` file.


## Sequence annotation
For prediction of attributes, please check our models in the ```prediction_models``` folder.

### command
To predict lab of origin based on input fasta file, please run the following command:
```Python
python prediction.py [-h] -m MODEL_DIR -i INPUT_FILE [-e] -nn NN_MODEL -l LAB_LIST [-o OUTPUT_FILE] [-n TOP_N]
```
The neural network model for lab prediction is provided in ```./prediction_models/embedding_prediction_labs.pth```. The lab labels are provided in ```./prediction_models/lab_list.txt```.

Arguments description

| argument | description |
| ------------- | ------------- |
| ```-h```, ```--help```  | show help message and exit  |
| ```-m```, ```--model_dir```  | path to the directory containing the pretrained model and tokenizer, __required__  |
|  ```-i```, ```--input_file```| FASTA file or embeddings file as input, __required__  |
| ```-e```, ```--embedding_file```| indicates if the input is an embedding file |
| ```-nn```, ```--nn_model```| path to the neural network model for lab prediction, __required__ |
| ```-l```, ```--lab_list```| path to the file containing the lab labels, __required__ |
| ```-o```, ```--output_file```| output file name for lab predictions |
| ```-n```, ```--top_n```| number of top predictions to output, default value: 10 |

The top predictions will be stored in the file ```lab_predictions.txt```, where each row corresponds to one input sequence.

### notebooks
We have provided the jupyter notebook [PlasmidGPT_predict.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebooks/PlasmidGPT_predict.ipynb) for the prediction of lab of origin.

The [Colab Notebook](https://colab.research.google.com/drive/1vo27RBnScf_cOISBdd13YN_hr5-ZVNHx?usp=sharing) can be easily used in the browser to predict the lab of origin, species, and vector type for the input sequence. The notebook will automatically download all related models and make predictions based on the user's input plasmid sequence. Please use the drop-down list to select the feature to predict, and the top 10 predictions will be displayed.


## Reference
- [PlasmidGPT: a generative framework for plasmid design and annotation](https://www.biorxiv.org/content/10.1101/2024.09.30.615762v1)
