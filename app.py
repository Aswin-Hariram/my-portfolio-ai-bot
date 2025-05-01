import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document  # Import Document

# Conversation imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)

def get_documents_from_pdf(pdf_path):
    """Loads and processes the PDF file to extract text."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # Split the extracted text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_text(text)

    # Convert each text chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in splitDocs]
    return documents


def create_db(docs):
    """Creates a vector store from the documents."""
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    """Creates a retriever chain that will be used for chat."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.9
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions using only the relevant information from the context below. Be clear and concise. Do not use phrases like 'Based on the information provided' or 'According to the context'. Just provide the answer directly.\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Set up the retriever to fetch relevant documents based on the question
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    """Process the chat with history to get the response."""
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response["answer"]


# Create documents and vector store once
pdf_path = 'Aswin_H_Developer_Portfolio.pdf'
docs = get_documents_from_pdf(pdf_path)
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

chat_history = []

# Enable CORS for the Flask app
CORS(app) 
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("input")
    if not user_input:
        return jsonify({"error": "Input is required"}), 400

    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False)
