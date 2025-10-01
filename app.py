import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

try:
    from google.colab import userdata
    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    os.environ["SERPAPI_API_KEY"] = userdata.get("SERPAPI_API_KEY")
    st.success("✅ API keys loaded from Colab Secrets")
except Exception as e:
    st.warning(f"⚠️ Could not load Colab secrets: {e}. Using .env fallback.")
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("SERPAPI_API_KEY"):
        st.error("❌ Missing GOOGLE_API_KEY or SERPAPI_API_KEY")
        st.stop()

@st.cache_resource
def process_pdf(pdf_file_path):
    st.info("Processing PDF... ⏳")
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.success("✅ PDF processed successfully")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return None

def get_agent_executor(vectorstore, llm, memory):
    pdf_retriever = vectorstore.as_retriever()
    pdf_tool = Tool(
        name="PDF Search",
        func=pdf_retriever.get_relevant_documents,
        description="Ask questions about the uploaded PDF."
    )

    try:
        serpapi = SerpAPIWrapper()
        google_tool = Tool(
            name="Google Search",
            func=serpapi.run,
            description="Search Google for external info."
        )
        tools = [pdf_tool, google_tool]
    except Exception as e:
        st.warning(f"⚠️ Could not init Google Search: {e}")
        tools = [pdf_tool]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")
st.title("🔬 AI Research Assistant (Gemini + Agent)")
st.markdown("Upload a PDF, ask questions, and let Gemini + Google Search help you.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_dir = "/content/temp_papers"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, uploaded_file.name)

    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        vectorstore = process_pdf(pdf_path)
        if vectorstore:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
            st.session_state.agent_executor = get_agent_executor(vectorstore, llm, st.session_state.memory)
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages.clear()
            st.session_state.memory.clear()
            st.success(f"📄 New PDF '{uploaded_file.name}' loaded")
        else:
            st.session_state.agent_executor = None
            st.session_state.last_file = None
else:
    st.info("⬆️ Upload a PDF to start")

if user_query := st.chat_input("Ask me something..."):
    if st.session_state.agent_executor:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_query})
                    output = response["output"]
                    st.markdown(output)
                    st.session_state.messages.append({"role": "assistant", "content": output})
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "I hit an error. Try again or rephrase."}
                    )
    else:
        st.warning("⚠️ Please upload a PDF first.")
