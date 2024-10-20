import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from Bio import SeqIO  # For reading FASTA files
import argparse
import os
import torch.nn as nn

# Function to validate DNA sequences (only A, T, C, G, N are allowed)
def validate_sequence(sequence):
    valid_characters = set("ATCGN")
    sequence_upper = sequence.upper()
    if not set(sequence_upper).issubset(valid_characters):
        raise ValueError(f"Invalid character(s) found in sequence: {sequence}")
    return sequence_upper

# Function to load model
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = torch.load(model_path)
    model.eval()
    model = model.to(device)
    
    return model

# Function to load tokenizer
def load_tokenizer(tokenizer_file):
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_file}' not found.")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    return tokenizer

# Function to calculate embeddings from DNA sequences
def calculate_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    for sequence in sequences:
        input_ids = tokenizer.encode(sequence.upper(), return_tensors='pt').to(device)
        model.config.output_hidden_states = True
        
        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs.hidden_states[-1].cpu().numpy()
            hidden_states_mean = np.mean(hidden_states, axis=1).reshape(-1)
            embeddings.append(hidden_states_mean)
    
    return np.array(embeddings)

# Function to read sequences from a FASTA file
def read_fasta_sequences(fasta_file):
    sequences = []
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file '{fasta_file}' not found.")
    
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            validated_sequence = validate_sequence(str(record.seq))  # Validate the DNA sequence
            sequences.append(validated_sequence)
    
    return sequences

# Neural Network model definition for lab prediction
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to load the neural network model for prediction
def load_nn_model(model_path, device, input_dim=768, num_classes=948):
    model_NN = SimpleNN(input_dim, num_classes).to(device)
    model_NN.load_state_dict(torch.load(model_path, map_location=device))
    model_NN.eval()
    return model_NN

# Function to perform prediction and get top N results
def predict_labs(embedding, model_NN, lab_list, top_n, device):
    X_data = torch.tensor([embedding], dtype=torch.float32).to(device)
    output = model_NN(X_data)
    
    # Calculate probabilities using softmax
    probs = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[0]
    
    # Get top N predictions
    prob_idx = np.argsort(probs)[::-1][:top_n]
    top_probs = probs[prob_idx]
    top_labels = lab_list[prob_idx]
    
    return top_labels, top_probs

# Main function to handle input arguments and processing
def main():
    parser = argparse.ArgumentParser(description="Predict lab of origin from DNA sequences.")
    
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Path to the pretrained model and tokenizer file.")
    parser.add_argument("-i","--input_file", type=str, required=True, help="FASTA file or embeddings file as input.")
    parser.add_argument("-e","--embedding_file", action='store_true', help="Indicates if the input is an embedding file.")
    parser.add_argument("-nn","--nn_model", type=str, required=True, help="Path to the neural network model for lab prediction.")
    parser.add_argument("-l","--lab_list", type=str, required=True, help="Path to the file containing the lab labels.")
    parser.add_argument("-o","--output_file", type=str, default="lab_predictions.txt", help="Output file for lab predictions.")
    parser.add_argument("-n","--top_n", type=int, default=10, help="Number of top predictions to output.")

    args = parser.parse_args()

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    # Load lab list
    with open(args.lab_list, 'r') as file:
        lab_list = np.array([line.strip() for line in file])

    # Load the neural network model for lab prediction
    model_NN = load_nn_model(args.nn_model, device)

    # If input is embeddings file
    if args.embedding_file:
        embeddings = np.loadtxt(args.input_file)
        if embeddings.shape == (768,):
            # Reshape the array to (1, 768)
            embeddings = embeddings.reshape(1, 768)
    else:
        # Load model and tokenizer
        model_path = os.path.join(args.model_dir, 'pretrained_model.pt')
        tokenizer_path = os.path.join(args.model_dir, 'addgene_trained_dna_tokenizer.json')
        model = load_model(model_path, device)
        tokenizer = load_tokenizer(tokenizer_path)
        
        # Read DNA sequences from the FASTA file and calculate embeddings
        sequences = read_fasta_sequences(args.input_file)
        embeddings = calculate_embeddings(model, tokenizer, sequences, device)

    # Predict labs and write results
    with open(args.output_file, 'w') as out_file:
        for idx, embedding in enumerate(embeddings):
            top_labels, top_probs = predict_labs(embedding, model_NN, lab_list, args.top_n, device)
            result = f"Sequence_{idx+1}\t" + "\t".join([f"{label}:{prob:.4f}" for label, prob in zip(top_labels, top_probs)]) + "\n"
            out_file.write(result)

    print(f"Lab predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()
