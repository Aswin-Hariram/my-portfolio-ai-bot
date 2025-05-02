from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Configuration ===
PDF_PATH = "Aswin_H_Developer_Portfolio.pdf"
CHROMA_DIR = "./chroma_db"
HUGGINGFACE_MODEL = "all-MiniLM-L6-v2"
CHROMA_K = 5
GITHUB_USERNAME = "Aswin-hariram"

# Get API keys from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Initialize Flask app ===
app = Flask(__name__)
CORS(app)

# Function to fetch GitHub profile data
def get_github_profile_data():
    user_url = f"https://api.github.com/users/{GITHUB_USERNAME}"
    repos_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    github_data = ""
    try:
        user_response = requests.get(user_url, headers=headers)
        if user_response.status_code == 200:
            user_data = user_response.json()
            github_data += f"GitHub Profile: {user_data.get('name', GITHUB_USERNAME)}\n"
            github_data += f"Bio: {user_data.get('bio', '')}\n"
            github_data += f"Location: {user_data.get('location', '')}\n"
            github_data += f"Public Repositories: {user_data.get('public_repos', 0)}\n\n"
        repos_response = requests.get(repos_url, headers=headers)
        if repos_response.status_code == 200:
            repos_data = repos_response.json()
            github_data += "Repositories:\n"
            for repo in repos_data:
                github_data += f"- {repo.get('name')}: {repo.get('description', 'No description')}\n"
                github_data += f"  Language: {repo.get('language', 'Not specified')}\n"
                github_data += f"  Stars: {repo.get('stargazers_count', 0)}\n"
                github_data += f"  URL: {repo.get('html_url')}\n\n"
    except Exception as e:
        print(f"Error fetching GitHub data: {str(e)}")
        github_data = f"GitHub profile for {GITHUB_USERNAME} (Error fetching detailed data)"
    return github_data

# === Pre-load model and vectorstore at startup ===
def initialize_qa_chain():
    global qa_chain
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        github_data = get_github_profile_data()
        github_doc = Document(page_content=github_data, metadata={"source": "GitHub Profile"})
        docs.append(github_doc)
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        # vectorstore.persist()  # REMOVE THIS LINE
        retriever = vectorstore.as_retriever(search_kwargs={"k": CHROMA_K})
        llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=OPENAI_API_KEY,
            model="openai/gpt-4.1",
            temperature=0.7
        )
        template = """Use the following pieces of context to answer the question at the end. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\nRespond directly to the question without using introductory phrases like 'Based on the available information' or other unnecessary phrases.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )
        print("✅ QA Chain is ready.")
    except Exception as e:
        raise RuntimeError(f"Startup failed: {str(e)}")

# Initialize on startup
qa_chain = None
initialize_qa_chain()

# === POST endpoint for querying ===
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"detail": "Question cannot be empty."}), 400
    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"question": question, "answer": result["result"]})
    except Exception as e:
        return jsonify({"detail": f"Error processing question: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
