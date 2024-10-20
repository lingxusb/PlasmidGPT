import torch
from transformers import PreTrainedTokenizerFast, GenerationConfig
import argparse
import os
from Bio import SeqIO  # Import for reading FASTA files

# Function to load model with error handling
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = torch.load(model_path)
    model.eval()
    model = model.to(device)
    
    return model

# Function to load tokenizer with error handling
def load_tokenizer(tokenizer_file):
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_file}' not found.")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    special_tokens_dict = {'additional_special_tokens': ['[PROMPT]', '[PROMPT2]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer

# Function to read the start sequence from a FASTA file
def read_fasta_sequence(fasta_file):
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file '{fasta_file}' not found.")
    
    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)  # Return the sequence from the first record in the file

    raise ValueError("No sequence found in the FASTA file.")

# Function to generate sequences
def generate_sequences(model, tokenizer, start_sequence, num_sequences, max_length, temperature, device):
    # Tokenize the start sequence and prepare input
    input_ids = tokenizer.encode(start_sequence.upper(), return_tensors='pt').to(device)
    
    # Add special tokens to the input
    special_tokens = torch.tensor([3] * 10 + [2], dtype=torch.long, device=device)
    input_ids = torch.cat((special_tokens.unsqueeze(0), input_ids), dim=1)
    
    # Sequence generation loop
    generated_sequences = []
    for j in range(num_sequences):
        while True:
            # Generate sequences with the model
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,  # Use the provided temperature for randomness
                do_sample=True,
                generation_config=GenerationConfig.from_model_config(model.config)
            )

            # Decode and clean the generated sequence
            generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
            if len(generated_sequence) > 200 + len(start_sequence):
                break
        
        print(f"Generated sequence {j+1}: {generated_sequence[0:50]}... (truncated)")
        generated_sequences.append(generated_sequence)
    
    return generated_sequences

# Function to write sequences to a FASTA file
def write_to_fasta(sequences, output_file):
    with open(output_file, 'w') as file:
        for idx, sequence in enumerate(sequences):
            file.write(f">PlasmidGPT_generate{idx+1}\n")
            file.write(sequence + "\n")
    print(f"Sequences written to {output_file}")

# Main function to handle input arguments and sequence generation
def main():
    # Argument parser for input arguments
    parser = argparse.ArgumentParser(description="Generate plasmid sequences using a pretrained model.")
    
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Path to the directory containing the pretrained model and tokenizer.")
    parser.add_argument("-s", "--start_sequence", type=str, help="Starting DNA sequence for sequence generation.")
    parser.add_argument("-f", "--fasta_file", type=str, help="FASTA file containing the starting sequence.")
    parser.add_argument("-n", "--num_sequences", type=int, default=1, help="Number of sequences to generate.")
    parser.add_argument("-l", "--max_length", type=int, default=300, help="Maximum length of the generated sequence.")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature for sequence generation (controls randomness).")
    parser.add_argument("-o", "--output", type=str, default="generated_sequence.fasta", help="Output file name for the generated sequences.")

    args = parser.parse_args()

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU.")

    # Load model and tokenizer
    model_path = os.path.join(args.model_dir, 'pretrained_model.pt')
    tokenizer_path = os.path.join(args.model_dir, 'addgene_trained_dna_tokenizer.json')
    
    try:
        model = load_model(model_path, device)
        tokenizer = load_tokenizer(tokenizer_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Determine the start sequence: from FASTA file or default
    if args.fasta_file:
        try:
            start_sequence = read_fasta_sequence(args.fasta_file)
            print(f"Using start sequence from FASTA file: {start_sequence[:50]}... (truncated)")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            return
    elif args.start_sequence:
        start_sequence = args.start_sequence
        print(f"Using provided start sequence: {start_sequence[:50]}... (truncated)")
    else:
        start_sequence = 'CCAATTATTGAAGGCCTCCCTAACGGGGGGCCTTTTTTTGTTTCTGGTCTCCCgcttGATAAGTCCCTAACTTTTACAGCTAGCTCAGTCCTAGGTATTATGCTAGCCTGAAGCTGTCACCGGATGTGCTTTCCGGTCTGATGAGTCCGTGAGGACGAAACAGCCTCTACAAATAATTTTGTTTAATACTAGAGAAAGAGGGGAAATACTAGATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACAGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTCGGCTACGGCCTGCAATGCTTCGCCCGCTACCCCGACCACATGAAGCTGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCAATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTTAGCTACCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA'
        print(f"Using default start sequence: {start_sequence[:50]}... (truncated)")

    # Generate sequences
    generated_sequences = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        start_sequence=start_sequence,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        temperature=args.temperature,  # Use the temperature argument
        device=device
    )

    # Write generated sequences to a FASTA file
    write_to_fasta(generated_sequences, args.output)

if __name__ == "__main__":
    main()
