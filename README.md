# PlasmidGPT: a generative framework for plasmid design and annotation
![github](https://github.com/user-attachments/assets/fc75bf4f-972c-4e3e-913e-499f01ab41ba)

We introduce PlasmidGPT, a generative language model pretrained on 153k engineered plasmid sequences from Addgene (https://www.addgene.org/). PlasmidGPT generates de novo sequences that share similar characteristics with engineered plasmids but show low sequence identity to the training data. We demonstrate its ability to generate plasmids in a controlled manner based on the input sequence or specific design constraint. Moreover, our model learns informative embeddings of both engineered and natural plasmids, allowing for efficient prediction of a wide range of sequence-related attributes.

### Trained model
The trained model and tokenizer is availale at [huggingface](https://huggingface.co/lingxusb/PlasmidGPT/tree/main). 

### Sequence generation
Please check our [Colab Notebook](https://colab.research.google.com/drive/1xWbekcTpcGMSiQE6LkRnqSTjswDkKAoc?usp=sharing) in the browser. Please make sure to connect to a GPU instance (e.g. T4 GPU).

### Sequence annotation
For the lab of origin prediction, please check our [Colab Notebook](https://colab.research.google.com/drive/1vo27RBnScf_cOISBdd13YN_hr5-ZVNHx?usp=sharing) in the browser.

### Reference
- [PlasmidGPT: a generative framework for plasmid design and annotation](https://www.biorxiv.org/content/10.1101/2024.09.30.615762v1)
