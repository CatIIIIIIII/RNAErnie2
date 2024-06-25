from time import time
from Bio import SeqIO


def filter_fasta(input_file, output_prefix):
    i = 0
    file_count = 1
    st_time = time()
    output_file = f"{output_prefix}_{file_count}.fasta"
    output_handle = open(output_file, 'w')

    for record in SeqIO.parse(input_file, "fasta"):
        SeqIO.write(record, output_handle, "fasta")
        i += 1
        if i % 5_000_000 == 0:
            output_handle.close()
            file_count += 1
            output_file = f"{output_prefix}_{file_count}.fasta"
            output_handle = open(output_file, 'w')
        if i % 1_000_000 == 0:
            print(f"Processed {i} sequences")
            print("Total time taken: ", time() - st_time)

    output_handle.close()
    print(f"Final processed sequences: {i}")
    print("Total time taken: ", time() - st_time)


# Example usage
input_file = "data/rnacentral/rnacentral_2048.fasta"
output_prefix = "split"
filter_fasta(input_file, output_prefix)
