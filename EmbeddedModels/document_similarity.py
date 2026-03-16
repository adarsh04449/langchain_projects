from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 300)

documents = [
    "Lionel Messi is known for his incredible dribbling, close ball control, and ability to score from almost any position on the field.",
    "Cristiano Ronaldo is famous for his athleticism, powerful shots, and remarkable goal-scoring record across multiple leagues.",
    "Kylian Mbappé stands out for his explosive speed and ability to break through defenses during counterattacks.",
    "Kevin De Bruyne is widely respected for his vision and precise passing that creates scoring opportunities for his teammates.",
    "Erling Haaland is recognized for his strength, positioning, and consistent ability to score goals inside the penalty box."
]

query = "tell me about haaland"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = (cosine_similarity([query_embedding], doc_embeddings))[0]

index, score = (sorted(list(enumerate(scores)), key = lambda x:x[1])[-1])

print(query)
print(documents[index])
print("similarity score is: ", score)