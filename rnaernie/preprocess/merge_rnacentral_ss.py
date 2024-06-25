from Bio import SeqIO

# List of input FASTA files
input_files = ["data/ss_1.fasta", "data/ss_2.fasta", "data/ss_3.fasta",
               "data/ss_4.fasta", "data/ss_5.fasta", "data/ss_6.fasta",
               "data/ss_7.fasta"]

# Output file
output_file = "data/merged.fasta"

# List to hold all sequences
all_sequences = []

# Read each input file and append sequences to the list
for file in input_files:
    with open(file, "r") as handle:
        sequences = list(SeqIO.parse(handle, "fasta"))
        all_sequences.extend(sequences)

# Write all sequences to the output file
with open(output_file, "w") as output_handle:
    SeqIO.write(all_sequences, output_handle, "fasta")

print(f"All sequences have been merged into {output_file}")
