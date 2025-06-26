import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "enter your key here"

# --- Load Vectorstore ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("gale_medical_index", embedding_model,allow_dangerous_deserialization=True)

# --- QA Chain ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo") 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="🩺 Doctor AI", layout="centered")
st.title("🩺 Medical AI Chatbot")
st.markdown("Ask me anything about medical related questions")

query = st.text_input("💬 Enter your medical question:")

if query:
    with st.spinner("Doctor AI is thinking..."):
        response = qa_chain(query)
        st.success("✅ Answer:")
        st.markdown(response["result"])

        with st.expander("📚 Source Excerpt"):
            for doc in response["source_documents"]:
                st.text(doc.page_content[:300])
