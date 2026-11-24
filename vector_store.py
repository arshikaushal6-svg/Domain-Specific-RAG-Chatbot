import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store(chunks):
    if not chunks: raise ValueError("There are no text chunks to index.")
    model = SentenceTransformer(MODEL_NAME)
    texts = [item["text"] for item in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"index":index, "model":model, "chunks":chunks}

def search_vector_store(question, store, top_k=4):
    query = store["model"].encode([question], convert_to_numpy=True, normalize_embeddings=True)
    query = np.asarray(query, dtype="float32")
    scores, ids = store["index"].search(query, min(top_k, len(store["chunks"])))
    found = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(store["chunks"][int(idx)])
            item["similarity"] = float(score)
            found.append(item)
    return found
