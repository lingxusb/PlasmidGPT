# PlasmidGPT: a generative framework for plasmid design and annotation
![github](https://github.com/user-attachments/assets/fc75bf4f-972c-4e3e-913e-499f01ab41ba)

We introduce PlasmidGPT, a generative language model pretrained on 153k engineered plasmid sequences from Addgene (https://www.addgene.org/). PlasmidGPT generates de novo sequences that share similar characteristics with engineered plasmids but show low sequence identity to the training data. We demonstrate its ability to generate plasmids in a controlled manner based on the input sequence or specific design constraint. Moreover, our model learns informative embeddings of both engineered and natural plasmids, allowing for efficient prediction of a wide range of sequence-related attributes.

### Install
To install `PlasmidGPT`, run the following bash script:
 ```bash
 git clone https://github.com/lingxusb/PlasmidGPT.git
 cd PlasmidGPT
 pip install .
```

### Trained model
The trained model and tokenizer is availale at [huggingface](https://huggingface.co/lingxusb/PlasmidGPT/tree/main). 

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
generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
```

Please check our jupyter notebook [PlasmidGPT_generate.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebook/PlasmidGPT_generate.ipynb). Or you can easily use our [Colab Notebook](https://colab.research.google.com/drive/1xWbekcTpcGMSiQE6LkRnqSTjswDkKAoc?usp=sharing) in the browser. Please make sure to connect to a GPU instance (e.g. T4 GPU).

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

We have provided the jupyter notebook: [PlasmidGPT_predict.ipynb](https://github.com/lingxusb/PlasmidGPT/blob/main/notebook/PlasmidGPT_predict.ipynb) and the
[Colab Notebook](https://colab.research.google.com/drive/1vo27RBnScf_cOISBdd13YN_hr5-ZVNHx?usp=sharing) which can be easily used in the browser.

### Reference
- [PlasmidGPT: a generative framework for plasmid design and annotation](https://www.biorxiv.org/content/10.1101/2024.09.30.615762v1)
