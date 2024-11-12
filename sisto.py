import streamlit as st
import random
import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

st.set_page_config("Hamza Ali Khan", page_icon="🙋")

st.title("Custom Gpt With Groq LLM")

# Define a list of motivational messages
motivational_messages = [
    "Set your morning routine and tackle procrastination.",
    "Believe in yourself! Start making progress today.",
    "Take small steps to reach your big goals."
]

# Display a random motivational message only if the user hasn't entered a message
if "user_entered" not in st.session_state:
    st.session_state.user_entered = False

if not st.session_state.user_entered:
    st.write(random.choice(motivational_messages))

st.sidebar.title("CHAT MESSAGES")
st.sidebar.text("Enter Text Length")

# Add file uploader in the sidebar
file = st.sidebar.file_uploader("Upload Your File", type="pdf")

def RAG():
    if file is not None:
        # Display loading message
        with st.spinner("Loading... ⌛"):
            doc_list = []

            # Read the PDF using PyMuPDF (fitz)
            pdf_document = fitz.open("pdf", file.read())  # file.read() gives the bytes
            r_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")  # Extract text content

                pg_chunks = r_splitter.split_text(page_text)
                for sub_pg in pg_chunks:
                    metadata = {"source": "RAG", "page_no": page_num + 1}
                    doc_store = Document(page_content=sub_pg, metadata=metadata)
                    doc_list.append(doc_store)

            pdf_document.close()
            st.success("PDF is loaded! Ready to use now 🎉")  # Success message
            return doc_list
    else:
        st.info("Please upload a PDF file to proceed.")
        return []

# Only initialize retriever if there is a document to retrieve from
docs = RAG()
retriever = TFIDFRetriever.from_documents(docs) if docs else None

llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.2-3b-preview"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response(query, chat_history):
    template = """
    You are a helpful assistant. Answer the following questions from given context:

    Context: {Context}

    User question: {Input}

    also consider the chat_history: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    qf = itemgetter("Input")
    cf = itemgetter("chat_history")

    setup = {"Input": qf, "chat_history": cf, "Context": qf | retriever | format_docs}

    chain = setup | prompt | llm | StrOutputParser()
    response = chain.invoke({"Input": query, "chat_history": chat_history})
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Enter Your Message:")

# Update the chat history and display response if there's user input
if user_input:
    st.session_state.user_entered = True  # Hide motivational text after input
    st.session_state.chat_history.append(HumanMessage(user_input))
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    if retriever:  # Ensure retriever is initialized
        with st.chat_message("assistant"):
            res = get_response(user_input, st.session_state.chat_history)
            st.markdown(res)
        
        st.session_state.chat_history.append(AIMessage(res))
    else:
        st.error("Please upload a valid PDF file to start the chat.")
