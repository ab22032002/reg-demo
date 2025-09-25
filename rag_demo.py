# 1.Handling Imports
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# 2.Module2
# Filepath (replacing with user input or upload mechanism)
# file_path = "my_document.pdf"
file_path = input("Enter the file path: ")
# print(f" entered path: {file_path}")

try:
    with open(file_path, 'r') as f:
        content = f.read()
        print("File content (first 100 chars):", content[:100])
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Choosing  loader based on file extension
if file_path.endswith(".txt"):
    loader = TextLoader(file_path)
elif file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    raise ValueError("Unsupported file format!")

documents = loader.load()

# 3. Chunk text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 4. Embeddings + Vector Store (for smart/efficient Searching)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.from_texts(docs, embeddings)
db = FAISS.from_documents(docs, embeddings)

# 5. Defining a Small Model (SLM)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 6. Retrieval + Q&A

while True:
    query = input("Ask your question (or type 'exit' to quit): ")
    
    if query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    
    docs = db.similarity_search(query, k=3)

    context = " ".join([d.page_content for d in docs])
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = llm(prompt)
    print("AI Answer:", response)

print("ThankYou")