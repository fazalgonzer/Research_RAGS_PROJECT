from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os 
import streamlit as st 
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if not (os.path.exists('faiss_index')):
        print("creating new DB its going to take time ")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        pdf_directory = "data/data2"

        loader = DirectoryLoader(path=pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)


        documents = loader.load()

        spliter2=RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
        docs=spliter2.split_documents(documents)

        uuids = [str(uuid4()) for _ in range(len(docs))]


        vector_store.add_documents(documents=docs, ids=uuids)
else:
    new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
query="Where is Fzal"

result = new_vector_store.similarity_search(query)
s=result[0].page_content
print(result)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
query_embedding = model.encode([query])
r_embedding=model.encode([s])
similarities = cosine_similarity(query_embedding,r_embedding)
print(similarities)

