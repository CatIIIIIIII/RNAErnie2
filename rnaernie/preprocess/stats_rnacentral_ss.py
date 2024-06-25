import re
import pickle
from time import time
from Bio import SeqIO
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_score(sequence):
    pattern = r'\(([-+]?[0-9]*\.?[0-9]+)\)$'
    match = re.search(pattern, sequence)

    if match:
        # Extract the score
        score = float(match.group(1))
        return score
    else:
        raise ValueError("No score found in the sequence")


def filter_fasta(input_file):
    SEQ_NUM = 3_000_000
    i = 0
    name_scores = {}

    st_time = time()
    with tqdm(total=SEQ_NUM) as pbar:
        for record in SeqIO.parse(input_file, "fasta"):
            pbar.update(1)
            i += 1
            name = record.id
            seq = str(record.seq)
            score = extract_score(seq)
            name_scores[name] = score
            # extract the last score from the sequence
            if i % SEQ_NUM == 0:
                print(f"Processed {i} sequences")
                print("Total time taken: ", time() - st_time)
                break

    print(f"Final processed sequences: {i}")
    print("Total time taken: ", time() - st_time)

    with open("preprocess/name_scores.pkl", "wb") as f:
        pickle.dump(name_scores, f)


# Example usage
input_file = "data/rnacentral_ss/rnacentral_ss_2048.fasta"
# filter_fasta(input_file)

name_scores = pickle.load(open("preprocess/name_scores.pkl", "rb"))
scores = list(name_scores.values())

# Plot the distribution of scores
plt.figure(figsize=(8, 6))
plt.hist(scores, bins=100, color='blue', edgecolor='black')
plt.title("Distribution of scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.savefig("preprocess/scores_distribution.png")

print(len([s for s in scores if s > -5]))
