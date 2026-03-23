from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

player1 = Document(
    page_content="Erling Haaland is a Norwegian striker known for his strength, speed, and incredible goal-scoring ability. He is one of the most dominant forwards in modern football.",
    metadata={"football_club": "Manchester City"}
)

player2 = Document(
    page_content="Kevin De Bruyne is a Belgian midfielder famous for his vision, precise passing, and creativity. He plays a key role in controlling the tempo of the game.",
    metadata={"football_club": "Manchester City"}
)

player3 = Document(
    page_content="Bukayo Saka is an English winger known for his dribbling, versatility, and consistency. He has become a crucial player for both club and country.",
    metadata={"football_club": "Arsenal"}
)

player4 = Document(
    page_content="Jude Bellingham is an English midfielder recognized for his maturity, technical skills, and leadership on the field. He is considered one of the brightest young talents in football.",
    metadata={"football_club": "Real Madrid"}
)

player5 = Document(
    page_content="Vinicius Junior is a Brazilian winger known for his pace, flair, and attacking creativity. He is a constant threat to defenders with his one-on-one ability.",
    metadata={"football_club": "Real Madrid"}
)

docs = [player1, player2, player3, player4, player5]

vector_store = Chroma(
    collection_name="sample",
    persist_directory="chroma_db",
    embedding_function=OpenAIEmbeddings()
)

result = vector_store.similarity_search(
    query = '',
    filter = {"football_club":"Real Madrid"}
)

print(result)
    
