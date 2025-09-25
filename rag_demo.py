# 1.Handling Imports
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 2.adding JD of a compmany for Quesry and Answers 
with open("sample.txt", "w") as f:
    f.write("Accenture focuses on SLMs, Agentic AI, and RAG pipelines for enterprise use cases.")

with open("sample.txt") as f:
    text = f.read()

# 3. Chunk text
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_text(text)

# 4. Embeddings + Vector Store (for smart/efficient Searching)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(docs, embeddings)

# 5. Defining a Small Model (SLM)
qa_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 6. Retrieval + Q&A
query = "What does Accenture focus on?"
docs = db.similarity_search(query, k=2)

context = " ".join([d.page_content for d in docs])
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

print(llm(prompt))
