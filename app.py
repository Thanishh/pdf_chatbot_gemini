import streamlit as st
import google.generativeai as genai
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import fitz
import csv

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to extract text from PDF files in a folder
def extract_text_from_pdf_folder(folder_path):
    all_text = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            all_text += text + '\n\n'  # Add a newline between documents
    return all_text

# Initialize Streamlit app
st.title("Chat with Gemini AI with Custom Data for an uploaded Document")

# File uploader for PDF documents
uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Initialize the model and QA chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, temperature=0.3, google_api_key='AIzaSyBqW4jdv4321epULq1mQaM9XCmvhVRSESU')
    chain = load_qa_chain(model, chain_type="stuff")  # Assuming load_qa_chain is defined elsewhere

    # Extract text from uploaded PDF files
    extracted_text = ""
    for uploaded_file in uploaded_files:
        extracted_text += extract_text_from_pdf(uploaded_file) + "\n\n"

    # Display extracted text
    st.text_area("Extracted Text", value=extracted_text, height=400)

    # Create vector store
    texts = [extracted_text]  # Use extracted text for vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyBqW4jdv4321epULq1mQaM9XCmvhVRSESU')  # Use provided embeddings model
    vector_store = Chroma.from_texts(texts, embeddings).as_retriever()

    # User input for questions
    questions = st.text_input("Enter your questions separated by commas")

    if st.button("Generate Responses"):
        if questions:
            question_responses = []

            # Iterate through each question
            for question in questions.split(','):
                # Get relevant documents
                docs = vector_store.get_relevant_documents(question)

                # Generate response
                response = chain({
                    "input_documents": docs,
                    "question": question.strip()
                }, return_only_outputs=True)

                # Append question-response pair to the list
                question_responses.append((question.strip(), response))

            # Display question-response pairs
            st.write("Question-Response Pairs:")
            for pair in question_responses:
                st.write(pair)

            # Save question-response pairs to a CSV file
            output_file = 'question_responses.csv'
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Question', 'Response'])
                writer.writerows(question_responses)

            st.write(f"Question-response pairs saved to '{output_file}'.")
        else:
            st.warning("Please enter at least one question.")
