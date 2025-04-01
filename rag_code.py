2. RAG 코드 공유드립니다.

import pandas as pd
import numpy as np
import faiss
from openai import OpenAI

# Initialize the OpenAI client (using Upstage settings)
client = OpenAI(
    api_key="up_GuigAOl5bgZmOGMDD3YzlJ5DRSk4Q",
    base_url="https://api.upstage.ai/v1"
)

# 1. Import the CSV file containing the chunks
df = pd.read_csv("D:/STUDY/2025-1/AGI agent Hackerthon/RAG_HTN_only.csv")
chunks = df["chunk_text"].tolist()

# 2. Convert all chunk texts into embeddings using the Upstage 'solar-embedding' model
# (Assuming the "solar-embedding" model is used for passage embedding)
embedding_response = client.embeddings.create(
    model="embedding-passage",
    input=chunks
).data

# Extract embeddings from the response
chunk_embeddings = [item.embedding for item in embedding_response]

# Convert the list of embeddings to a NumPy array (float32 is required by FAISS)
embeddings_np = np.array(chunk_embeddings).astype("float32")

# Determine the embedding dimension from the first embedding
d = embeddings_np.shape[1]

# 3. Create a FAISS index (using L2 distance) and add the embeddings to it
index = faiss.IndexFlatL2(d)
index.add(embeddings_np)
print("Number of vectors in FAISS index:", index.ntotal)

# Optionally, save the FAISS index to disk for later use
faiss.write_index(index, "D:/STUDY/2025-1/AGI agent Hackerthon/HTN_faiss_index.bin")


# 4. Test the FAISS index by querying with a sample query
query_text = "혈압 높다고 뭐가 대수인가요? 왜 이렇게 유난인지..."
# Generate the query embedding using the 'embedding-query' model
query_response = client.embeddings.create(
    model="embedding-query",
    input=query_text
)
# Access the embedding based on the response structure
if hasattr(query_response, 'data'):
    query_embedding = np.array([query_response.data[0].embedding]).astype("float32")
else:
    # If query_response is already a list
    query_embedding = np.array([query_response[0].embedding]).astype("float32")

# Search for the top 3 most similar chunks in the FAISS index
k = 3
D, I = index.search(query_embedding, k)

print("\nTop 3 matched chunk indices and texts:")
for idx in I[0]:
    row = df.iloc[idx]
    print(f"Chunk ID: {row['chunk_id']} | Category: {row['chunk_category']}")
    print(row['chunk_text'])
    print("-" * 50)
