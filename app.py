from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

app = Flask(__name__)
import nltk
nltk.download('punkt_tab')

pdf_path = "Updated Remark App Description.pdf"
google_api_key = 'AIzaSyAJLv_QjBn1QPliUJ6_CTR4peHzd2cXVYg'
embedding_model_path = "models/embedding-001"

def initialize_chatbot(pdf_path, google_api_key, embedding_model_path):
    model = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash-latest',
        temperature=0.7,
        google_api_key=google_api_key
    )
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=embedding_model_path,
        google_api_key=google_api_key
    )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    doc_search = DocArrayInMemorySearch.from_documents(splits, embedding_model)

    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=doc_search.as_retriever(),
        chain_type="stuff"  
    )
    
    return rag_chain

rag_chain = initialize_chatbot(pdf_path, google_api_key, embedding_model_path)

def chat_with_remark(rag_chain, user_input):
    response = rag_chain.run(query=user_input)
    response_text = ' '.join(response.split())  # Removing unwanted spaces or newlines
    return [response_text]

@app.route('/chatbot', methods=['GET'])
def chatbot():
    user_input = request.args.get('q', '')
    if user_input:
        response = chat_with_remark(rag_chain, user_input)
        return jsonify({'response': response[0]})
    return jsonify({'error': 'No query parameter provided'}), 400

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
