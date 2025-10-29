# generate a faiss index for the language embeddings
# cosine similarity with normalization
import argparse, numpy as np, faiss
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", default="../outputs/language_embeddings.npz")
    ap.add_argument("--out_index", default="../outputs/faiss_polybert_cosine.index")
    return ap.parse_args()
def main():
    a = parse()
    D = np.load(a.emb_npz, allow_pickle=True)
    Z = D["z_text"].astype("float32")
    # Cosine via L2-normalize + inner-product index
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Z = Z / norms
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(Z)
    faiss.write_index(index, a.out_index)
    print("Index built:", a.out_index, "size:", index.ntotal)
    
if name == "main":
    main()