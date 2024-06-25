import os.path as osp
import pickle
import os

embeddings = {}
for i in range(8):
    with open(osp.join(f"outputs/inference/bprna/embeddings{i}.pickle"), "rb") as f:
        embeddings.update(pickle.load(f))
print(embeddings[0].shape)

with open(osp.join("outputs/inference/bprna/bprna_embeddings.pickle"), "wb") as f:
    pickle.dump(embeddings, f)
# for i in range(8):
#     os.remove(osp.join(f"outputs/inference/rfam_crw/embeddings{i}.pickle"))

with open(osp.join("outputs/inference/bprna/bprna_embeddings.pickle"), "rb") as f:
    embeddings = pickle.load(f)

print(embeddings[0].shape)
