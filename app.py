from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import requests
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

PDF_PATH = "Aswin_H_Developer_Portfolio.pdf"
CHROMA_DIR = "./chroma_db"
HUGGINGFACE_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # lighter model
CHROMA_K = 5
GITHUB_USERNAME = "Aswin-hariram"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="PDF QA API", version="1.0")

# === CORS middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

def get_github_profile_data():
    user_url = f"https://api.github.com/users/{GITHUB_USERNAME}"
    repos_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    github_data = ""
    try:
        user_res = requests.get(user_url, headers=headers)
        if user_res.ok:
            user = user_res.json()
            github_data += f"GitHub Profile: {user.get('name', GITHUB_USERNAME)}\n"
            github_data += f"Bio: {user.get('bio', '')}\n"
            github_data += f"Location: {user.get('location', '')}\n"
            github_data += f"Public Repositories: {user.get('public_repos', 0)}\n\n"

        repo_res = requests.get(repos_url, headers=headers)
        if repo_res.ok:
            repos = repo_res.json()[:5]  # limit to 5 repos
            github_data += "Repositories:\n"
            for repo in repos:
                github_data += f"- {repo.get('name')}: {repo.get('description', 'No description')}\n"
                github_data += f"  Language: {repo.get('language', 'Not specified')}, Stars: {repo.get('stargazers_count', 0)}\n"
                github_data += f"  URL: {repo.get('html_url')}\n\n"
    except Exception as e:
        github_data = f"GitHub profile for {GITHUB_USERNAME} (Error fetching detailed data)"

    return github_data

@app.on_event("startup")
def startup_event():
    global qa_chain
    try:
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)

        # Load vectorstore if already exists
        if os.path.exists(os.path.join(CHROMA_DIR, "index")):
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            github_text = get_github_profile_data()
            github_doc = Document(page_content=github_text, metadata={"source": "GitHub Profile"})
            docs.append(github_doc)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
            vectorstore.persist()

        retriever = vectorstore.as_retriever(search_kwargs={"k": CHROMA_K})

        llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=OPENAI_API_KEY,
            model="openai/gpt-4.1",
            temperature=0.7
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        print("✅ QA Chain is ready.")
    except Exception as e:
        raise RuntimeError(f"Startup failed: {str(e)}")

@app.post("/ask")
async def ask_question(query: QueryRequest):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"question": query.question, "answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
