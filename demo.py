import streamlit as st
import pandas as pd
import json
import os
import re
import time
import shutil
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import plotly.graph_objects as go
from dotenv import load_dotenv
import isodate

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# ------------------- CONSTANTS -------------------
CHROMA_DB_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.3

# ------------------- CONFIGURATION -------------------
def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    return {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    default_values = {
        'video_loaded': False,
        'transcript': None,
        'video_id': None,
        'video_title': None,
        'chat_history': [],
        'recommendations': [],
        'analysis_summary': "",
        'api_keys_set': False,
        'last_query': "",
        'repeat_count': 0,
        'qa_chain': None,
        'show_recommendations': False,
        'retriever': None
    }
    
    for key, default in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="YouTube Q&A Assistant",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ------------------- YOUTUBE FUNCTIONALITY -------------------
def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return match.group(6) if match else None

def get_transcript(video_id):
    """Get transcript for a YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        timestamp_data = [{'time': entry['start'], 'text': entry['text']} for entry in transcript_list]
        return transcript_text, timestamp_data
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None, None

def get_video_details(video_id, youtube_api_key):
    """Get video details from YouTube API"""
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        request = youtube.videos().list(part="snippet,contentDetails,statistics", id=video_id)
        response = request.execute()
        if not response['items']:
            return None
        video_data = response['items'][0]['snippet']
        return {
            'title': video_data['title'],
            'description': video_data['description'],
            'channel': video_data['channelTitle'],
            'published_at': video_data['publishedAt'],
            'thumbnail': video_data['thumbnails']['high']['url'],
            'views': response['items'][0]['statistics'].get('viewCount', '0'),
            'likes': response['items'][0]['statistics'].get('likeCount', '0'),
            'duration': response['items'][0]['contentDetails'].get('duration', 'PT0S')
        }
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None

# ------------------- LLM FUNCTIONALITY -------------------
class LLMYouTubeRecommender:
    """Class to handle YouTube video recommendations using LLM"""
    
    def __init__(self, youtube_api_key, openai_api_key):
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=openai_api_key)
        
    def get_llm_based_recommendations(self, video_id, transcript, video_details):
        """Generate recommendations based on video content using LLM"""
        # First, use LLM to analyze the content and suggest keywords/topics
        analysis_prompt = f"""
        Analyze this YouTube video transcript and extract 5-7 key topics or keywords that best represent the content.
        
        Video Title: {video_details['title']}
        Video Description: {video_details['description']}
        
        Transcript excerpt (first 2000 chars): {transcript[:2000]}...
        
        Format your response as a JSON with:
        1. "topics": A list of topics/keywords
        2. "summary": A brief content summary (max 3 sentences)
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            analysis = json.loads(response.content)
            
            # Use these topics to search for related videos
            search_query = " OR ".join(analysis["topics"][:5])  # Use top 5 topics
            
            search_response = self.youtube.search().list(
                q=search_query,
                part="snippet",
                type="video",
                maxResults=10,
                relevanceLanguage="en",
                videoDefinition="high"
            ).execute()
            
            recommendations = []
            for item in search_response.get("items", []):
                if item["id"]["videoId"] != video_id:  # Exclude the original video
                    recommendations.append({
                        "video_id": item["id"]["videoId"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                        "description": item["snippet"]["description"]
                    })
            
            return recommendations, analysis["summary"]
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return [], "Could not generate recommendations due to an error."

def setup_qa_chain():
    """Set up the LangChain conversational retrieval chain"""
    if not st.session_state.transcript:
        st.error("No transcript available to analyze. Please load a video first.")
        return False
    
    try:
        transcript_text = st.session_state.transcript
        documents = [Document(page_content=transcript_text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
        vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DB_DIRECTORY)
        st.session_state.retriever = vectordb.as_retriever()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=st.session_state.openai_api_key)

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.retriever,
            memory=memory
        )
        return True
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return False

def query_llm_with_memory(user_input):
    """Query the LLM with the user input and context from memory"""
    # First check if API keys are set
    if not st.session_state.api_keys_set:
        return "Please set your API keys in the sidebar first."
    
    # Then check if a video is loaded
    if not st.session_state.video_loaded or not st.session_state.transcript:
        return "Please load a YouTube video first."
    
    # Then check if QA chain exists, if not set it up
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        setup_success = setup_qa_chain()
        if not setup_success:
            return "Failed to initialize the question answering system. Please check your API keys and try again."
    
    # Now try querying
    try:
        result = st.session_state.qa_chain({"question": user_input})
        return result["answer"]
    except Exception as e:
        return f"Error: {str(e)}. Please try reloading the video or check your API keys."

def clear_conversation():
    """Clear the conversation history and delete the vector database"""
    # Clear session state
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.retriever = None
    st.session_state.last_query = ""
    st.session_state.repeat_count = 0
    st.session_state.recommendations = []
    st.session_state.analysis_summary = ""
    st.session_state.show_recommendations = False
    
    # Delete vector database directory
    if os.path.exists(CHROMA_DB_DIRECTORY):
        try:
            shutil.rmtree(CHROMA_DB_DIRECTORY)
            return True
        except Exception as e:
            st.error(f"Error clearing vector database: {str(e)}")
            return False
    return True

def filter_out_shorts(recommendations, youtube_api_key):
    """Filter out YouTube Shorts (videos shorter than 60 seconds)"""
    filtered_recs = []
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    for video in recommendations:
        try:
            vid_id = video['video_id']
            details = youtube.videos().list(
                part='contentDetails',
                id=vid_id
            ).execute()

            if details['items']:
                duration = details['items'][0]['contentDetails']['duration']
                seconds = isodate.parse_duration(duration).total_seconds()
                if seconds >= 60:
                    filtered_recs.append(video)
            else:
                # Keep if duration info not available
                filtered_recs.append(video)
        except Exception:
            # Keep on error
            filtered_recs.append(video)
            
    return filtered_recs

# ------------------- UI COMPONENTS -------------------
def render_sidebar():
    """Render the sidebar with API key inputs"""
    with st.sidebar:
        st.title("🔑 API Keys")
        admin_mode = st.checkbox("👑 Admin Mode (Use .env keys)")
        
        env_vars = load_environment()
        
        if admin_mode:
            if env_vars["youtube_api_key"] and env_vars["openai_api_key"]:
                st.success("✅ Using API keys from environment variables!")
                st.session_state.youtube_api_key = env_vars["youtube_api_key"]
                st.session_state.openai_api_key = env_vars["openai_api_key"]
                st.session_state.api_keys_set = True
            else:
                st.error("❌ API keys not found in environment variables!")
        else:
            youtube_api_key_input = st.text_input("YouTube API Key", type="password")
            openai_api_key_input = st.text_input("OpenAI API Key", type="password")
            if st.button("Save API Keys"):
                if youtube_api_key_input and openai_api_key_input:
                    st.session_state.youtube_api_key = youtube_api_key_input
                    st.session_state.openai_api_key = openai_api_key_input
                    st.session_state.api_keys_set = True
                    st.success("API keys saved successfully!")
                else:
                    st.error("Both API keys are required")
        
        # Add help information in sidebar
        st.markdown("---")
        st.markdown("### 🤖 Help")
        st.markdown("""
        - Type questions about the video content
        - Type `/relate` to get related videos
        - Click "Clear Conversation" to start fresh
        """)

def render_video_input():
    """Render the video URL input and load button"""
    st.title("🎬 YouTube Q&A and Recommendation Assistant")
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input("Enter YouTube Video URL", key="video_url_input")
    with col2:
        load_video_button = st.button("Load Video")
        
    return video_url, load_video_button

def render_video_player():
    """Render the video player and details"""
    if st.session_state.video_loaded:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        with col2:
            st.markdown(f"### {st.session_state.video_title}")
            if 'video_details' in st.session_state:
                details = st.session_state.video_details
                st.markdown(f"**Channel:** {details['channel']}")
                st.markdown(f"**Views:** {int(details['views']):,}")
                st.markdown(f"**Likes:** {int(details['likes']):,}")
                st.markdown("---")
              

def render_message(message):
    """Render a single chat message with proper styling"""
    if message['role'] == 'user':
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                <div style="
                    background: linear-gradient(135deg, #daf3ff, #bee3f8);
                    color: #003366;
                    padding: 12px 16px;
                    border-radius: 16px;
                    max-width: 75%;
                    font-size: 15px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                ">
                    💬 <strong>You:</strong> {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="
                    background: linear-gradient(135deg, #f4f4f4, #eeeeee);
                    color: #333333;
                    padding: 12px 16px;
                    border-radius: 16px;
                    max-width: 75%;
                    font-size: 15px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                ">
                    🤖 <strong>Assistant:</strong> {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

def render_recommendations():
    """Render video recommendations with nice styling"""
    if st.session_state.recommendations and st.session_state.show_recommendations:
        st.markdown("""
        <div style="background-color: #FFBF00;color:#36454F; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="margin-top: 0;">🎥 Related Videos</h3>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Content Analysis:** {st.session_state.analysis_summary}")
        
        # Display recommendations in a grid of 3 columns
        cols = st.columns(3)
        for i, rec in enumerate(st.session_state.recommendations[:6]):  # Show up to 6 recommendations
            with cols[i % 3]:
                st.image(rec["thumbnail"], use_container_width =True)
                st.markdown(f"**[{rec['title']}](https://www.youtube.com/watch?v={rec['video_id']})**")
                st.caption(f"Channel: {rec['channel']}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_chat_interface():
    """Render the chat interface with history and input form"""
    st.markdown("### 💬 Chat with Your Video")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    # Create a container for the input form at the bottom
    input_container = st.container()
    
    # Display chat history in the chat container first
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Ask a question about the video content to get started!")
        else:
            for message in st.session_state.chat_history:
                render_message(message)
                # If this was a /relate command, show recommendations after the assistant's response
                if message['role'] == 'user' and "/relate" in message['content'].lower():
                    st.session_state.show_recommendations = True
            
            # Show recommendations if needed (right after related videos request)
            render_recommendations()
    
    # Render chat input form at the bottom
    with input_container:
        with st.form("query_form", clear_on_submit=True):
            user_query = st.text_input("Ask a question or type /relate for similar videos", key="user_query")
            cols = st.columns(4)
            with cols[0]:
                submitted = st.form_submit_button("Send")
            with cols[1]:
                recommendation_button = st.form_submit_button("Recommendation")
            with cols[2]:
                clear_button = st.form_submit_button("🗑️ Clear")
            with cols[3]:
                help_button = st.form_submit_button("❓ Help")
            
    if recommendation_button:
        with st.spinner("Finding related videos using LLM analysis..."):
            try:
                recommender = LLMYouTubeRecommender(
                    youtube_api_key=st.session_state.youtube_api_key,
                    openai_api_key=st.session_state.openai_api_key
                )

                recommendations, summary = recommender.get_llm_based_recommendations(
                    st.session_state.video_id,
                    st.session_state.transcript,
                    st.session_state.video_details
                )

                # Filter out YouTube Shorts
                filtered_recs = filter_out_shorts(recommendations, st.session_state.youtube_api_key)

                st.session_state.recommendations = filtered_recs
                st.session_state.analysis_summary = summary
                st.session_state.show_recommendations = True
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "/relate (Find related videos)"
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Here are some related videos based on the content analysis."
                })
                return True
            except Exception as e:
                st.error(f"Error finding related videos: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "/relate (Find related videos)"
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error while finding related videos: {str(e)}"
                })
                return True
            
    # Process form submissions
    if clear_button:
        if clear_conversation():
            st.success("Conversation cleared!")
            st.rerun()
            
    if help_button:
        st.session_state.chat_history.append({
            "role": "user", 
            "content": "Show me help information"
        })
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": """
            Here's how to use this YouTube Q&A Assistant:
            
            - **Ask questions** about the video content and I'll analyze the transcript to answer
            - Type **/relate** to get recommendations for related videos
            - Use the **Clear** button to start a fresh conversation
            - You can ask about specific parts of the video, themes, people mentioned, etc.
            
            Try questions like:
            - "What is the main topic of this video?"
            - "Can you summarize the key points?"
            - "Who was mentioned at 3:45 in the video?"
            """
        })
        st.rerun()
    
    # Process user query
    if submitted and user_query:
        handle_user_query(user_query, True)
        st.rerun()

# ------------------- MAIN APPLICATION FLOW -------------------
def handle_video_loading(video_url, load_video_button):
    """Handle loading a YouTube video"""
    if load_video_button and video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            with st.spinner("Loading video information and transcript..."):
                if not st.session_state.api_keys_set:
                    st.error("Please set your API keys in the sidebar first.")
                    return False
                
                details = get_video_details(video_id, st.session_state.youtube_api_key)
                transcript, timestamps = get_transcript(video_id)
                
                if details and transcript:
                    # Clear previous conversation if loading a new video
                    clear_conversation()
                    
                    st.session_state.update({
                        "video_id": video_id,
                        "video_title": details['title'],
                        "transcript": transcript,
                        "timestamp_data": timestamps,
                        "video_details": details,
                        "video_loaded": True
                    })
                    st.success(f"Loaded: {details['title']}")
                    return True
                else:
                    st.error("Failed to fetch video details or transcript.")
                    return False
        else:
            st.error("Invalid YouTube URL")
            return False
    return False

def handle_user_query(user_query, submitted):
    """Process user query and generate response"""
    if not submitted or not user_query:
        return False
        
    user_query = user_query.strip()
    
    if not st.session_state.api_keys_set:
        st.error("Please set your API keys in the sidebar first.")
        return False
    
    # Handle special /relate command

        # Check for repeated questions
    if user_query.lower() == st.session_state.last_query:
            st.session_state.repeat_count += 1
            if st.session_state.repeat_count > 1:
                st.warning(f"You've asked this same question {st.session_state.repeat_count} times.")
                if st.session_state.repeat_count > 3:
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "You've asked this multiple times. Please try rephrasing your question or ask something different."
                    })
                    return True
    else:
        st.session_state.last_query = user_query.lower()
        st.session_state.repeat_count = 0

    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Analyzing transcript to answer your question..."):
        answer = query_llm_with_memory(user_query)
        if answer:
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            return True
        else:
            st.error("❌ Failed to get a response. Please try again.")
                # Add error message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Sorry, I couldn't process your question. Please try again or reload the video."
            })
            return True
    
    return False   

def main():
    """Main application function"""
    # Setup
    setup_page_config()
    initialize_session_state()
    render_sidebar()
    
    # Video input
    video_url, load_video_button = render_video_input()
    
    # Handle video loading
    if handle_video_loading(video_url, load_video_button):
        st.rerun()
    
    # Display video if loaded
    if st.session_state.video_loaded:
        # Create a two-column layout: video player on left, chat on right
        render_video_player()
        
        # Always show the chat interface when video is loaded
        render_chat_interface()

if __name__ == "__main__":
    main()