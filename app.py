import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
import tempfile


# Set API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configure the page
st.set_page_config(
    page_title="YouTube Video Q&A Bot",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "transcript_ready" not in st.session_state:
    st.session_state.transcript_ready = False

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "processing" not in st.session_state:
    st.session_state.processing = False

def video_to_transcript(url: str) -> str:
    """
    Extracts and returns the transcript text from a YouTube video URL.

    Parameters:
        url (str): YouTube video URL in the format containing '?v='.

    Returns:
        str: Full transcript text.
    """
    # Extract video ID using original split logic
    video_id = url.split("=")[1]

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {e}")

    # Join all text entries into a single string
    transcript_text = " ".join(entry['text'] for entry in transcript)

    return transcript_text

def build_qa_bot(transcript_text):
    """Build the QA bot with conversation memory"""
    # Save transcript to temp file
    temp_filename = "temp_transcript.txt"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    # Load and split text
    loader = TextLoader(temp_filename)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Generate embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    
    
    template = """You are a helpful assistant that provides detailed answers about YouTube video content.
    Use the following context to answer the question. If you don't know the answer based on the context, 
    explain what you do know but don't make up information.
    
    Always provide comprehensive, well-structured answers - never respond with just one line.
    Include specific details from the video whenever possible.
    
    Context: {context}
    
    Question: {question}
    
    Detailed Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Create conversational retrieval chain with memory
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=True
    )
    
    return qa_chain

def reset_conversation():
    """Reset the conversation and memory"""
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    st.session_state.transcript_ready = False
    if "qa_chain" in st.session_state:
        del st.session_state.qa_chain
    st.session_state.processing = False

def main():
    # Set up the sidebar
    with st.sidebar:
        st.title("📺 YouTube Q&A Bot")
        st.markdown("---")
        
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube Video URL", key="youtube_url")
        
        # Process video button
        process_button = st.button("Process Video 🔍", use_container_width=True)
        
        # Reset conversation button
        reset_button = st.button("Reset Conversation 🔄", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This bot extracts the transcript from a YouTube video and answers "
            "questions about its content using AI. It remembers your conversation "
            "history within the current session."
        )

    # Main content area - Chat interface
    st.title("🎥 YouTube Video Q&A Bot")
    
    # Process button logic
    if process_button and youtube_url:
        # Reset conversation when processing a new video
        st.session_state.messages = []
        st.session_state.processing = True
        
        # Add processing message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🔍 Extracting transcript from video..."
        })

    # Reset button logic
    if reset_button:
        reset_conversation()
    
    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Process video if needed
    if st.session_state.processing:
        # Extract transcript
        with st.spinner("Processing video..."):
            transcript = video_to_transcript(youtube_url)
            
            if transcript:
                # Add success message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ Transcript extracted successfully! Building QA system..."
                })
                
                # Build QA system
                qa_chain = build_qa_bot(transcript)
                st.session_state.qa_chain = qa_chain
                st.session_state.transcript_ready = True
                
                # Add ready message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🤖 I'm ready to answer questions about this video! Ask me anything."
                })
            else:
                # Add failure message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ Failed to extract transcript. Please check the URL and try again."
                })
            
            # Clear processing flag
            st.session_state.processing = False
            st.rerun()
    
    # Chat input
    if st.session_state.transcript_ready:
        user_question = st.chat_input("Ask a question about the video")
        
        if user_question:
            # Add user question to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })
            
            # Add thinking message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get answer from QA chain
                    response = st.session_state.qa_chain({"question": user_question})
                    answer = response["answer"]
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            
            # Rerun to update the UI
            st.rerun()
    else:
        st.info("👈 Enter a YouTube URL in the sidebar and click 'Process Video' to start.")

if __name__ == "__main__":
    main()