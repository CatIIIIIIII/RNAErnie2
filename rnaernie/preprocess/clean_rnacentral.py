from time import time
from Bio import SeqIO
import random

def is_valid_rna(sequence):
    """Checks if the RNA sequence contains only valid characters (A, C, U, G, T, N)."""
    valid_chars = set("ACUGTN")
    return all(char in valid_chars for char in sequence)

def filter_fasta(input_file):
    i = 0
    j = 0
    st_time = time()
    output_file = "rnacentral_shuffled.fasta"

    records = []
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) <= 2048 and is_valid_rna(sequence):
            records.append(record)
            i += 1
            if i % 1_000_000 == 0:
                print(f"Processed {i} sequences, ignored {j} sequences.")
                print("Total time taken: ", time() - st_time)
                # break
        else:
            j += 1

    # Shuffle the records
    random.shuffle(records)

    # Write the shuffled records to the output file
    with open(output_file, 'w') as output_handle:
        SeqIO.write(records, output_handle, "fasta")

    print(f"Final processed sequences: {i}, ignored sequences: {j}.")
    print("Total time taken: ", time() - st_time)

# Example usage
input_file = "/data/raw/rnacentral_active.fasta"
filter_fasta(input_file)
