# 🎥 YouTube Video Q\&A Bot 🤖

A **Streamlit** web app that extracts transcripts from YouTube videos and answers your questions about the video content using **OpenAI's GPT models** and **LangChain**.

---

## ✨ Features

* 📜 Extracts the transcript of any YouTube video via URL
* 🤖 Builds a conversational AI assistant to answer detailed questions based on the transcript
* 🔍 Uses OpenAI embeddings and vector search for fast, efficient document retrieval
* 🧠 Maintains conversation memory for contextual Q\&A
* ⏳ Shows progress bar and status updates during transcript extraction and vector processing
* 🖥️ Clean, user-friendly chat interface powered by Streamlit

---

## 🎬 Demo

![dashboard](dashboard.jpg)
*(Add your demo GIF or screenshot here)*

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.8+
* OpenAI API key (Get yours from [OpenAI](https://platform.openai.com/account/api-keys))
* Streamlit installed (`pip install streamlit`)
* Other dependencies (see below)

### 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Akshay-K8/YoutubeVideo_QA.git
   cd YoutubeVideo_QA
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Ensure the `extract_transcript_text` function in `Scraping.py` properly handles transcript extraction from YouTube URLs.

---

## 🏃 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* 🔗 Enter a YouTube video URL in the sidebar
* ▶️ Click **Process Video** to extract the transcript and build the QA system
* ❓ Ask any question about the video in the chat interface
* 🔄 Click **Reset Conversation** to start over

---

## 📂 Code Structure

* `app.py` — Main Streamlit app code with UI and logic
* `Scraping.py` — Contains the `extract_transcript_text` function for transcript scraping
* `.env` — Environment variables including OpenAI API key
* `requirements.txt` — Python dependencies

---

## 🛠️ Technologies Used

* [Streamlit](https://streamlit.io/) — Web app framework for Python
* [OpenAI GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5) — Conversational language model
* [LangChain](https://python.langchain.com/en/latest/) — Framework for building LLM-powered apps
* [Chroma](https://www.trychroma.com/) — Vector store for semantic search
* Custom YouTube transcript scraping logic

---

## ⚠️ Notes

* Transcript extraction depends on YouTube’s page structure and may need updates if YouTube changes its DOM.
* Uses OpenAI APIs which may incur costs.
* Designed primarily for educational and prototyping purposes.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

---

If you run into any issues, please don't hesitate to reach out! 😊

---

Would you like me to also help generate a cool project logo or demo GIF ideas?
