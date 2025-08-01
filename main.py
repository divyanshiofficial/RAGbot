import os
import pickle
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from PIL import Image
import faiss
import json
import numpy as np

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.0-flash')
text_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = genai

# text_model = genai.GenerativeModel('gemini-1.5-flash')
# vision_model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit setup
st.set_page_config(page_title="RAG Bot", layout="wide")
st.title("🧠 RAG Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.stored_chunks = []

# Sidebar uploads
st.sidebar.header("Upload Files")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
uploaded_images = st.sidebar.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

QUESTIONS_FILE = "questions.json"
SIMILARITY_THRESHOLD = 0.99  # Adjust as needed

def load_questions():
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_questions(questions):
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_question(user_question, questions):
    user_emb = embed_text(user_question)
    for q in questions:
        q_emb = embed_text(q["question"])  # Compute embedding on the fly
        sim = cosine_similarity(user_emb, q_emb)
        if sim >= SIMILARITY_THRESHOLD:
            return q["answer"]
    return None

# PDF processing
def extract_text_from_pdfs(pdf_files):
    combined_text = ""
    for uploaded_file in pdf_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"
    return combined_text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    response = embedding_model.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def store_chunks_and_embeddings(text):
    chunks = chunk_text(text)
    embeddings = [embed_text(chunk) for chunk in chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    st.session_state.stored_chunks = chunks
    st.session_state.faiss_index = index
    save_vector_db(index, chunks)  # Save to disk

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = np.array([embed_text(query)]).astype("float32")
    distances, indices = st.session_state.faiss_index.search(query_embedding, top_k)
    results = [st.session_state.stored_chunks[i] for i in indices[0]]
    return "\n".join(results)

# Vector DB persistence
def save_vector_db(index, chunks, path="faiss_index.index", chunks_path="chunks.pkl"):
    faiss.write_index(index, path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_db(path="faiss_index.index", chunks_path="chunks.pkl"):
    if os.path.exists(path) and os.path.exists(chunks_path):
        index = faiss.read_index(path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        st.session_state.faiss_index = index
        st.session_state.stored_chunks = chunks

# Try to load existing vector DB at startup
load_vector_db()

# Image processing
def extract_insight_from_images(image_files):
    all_captions = ""
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            response = vision_model.generate_content(["Describe this image in detail", img])
            caption = response.text.strip()
            all_captions += f"\nImage ({img_file.name}): {caption}\n"
        except Exception as e:
            all_captions += f"\nImage ({img_file.name}): ❌ Error - {str(e)}\n"
    return all_captions

# Q&A function
def ask_combined_question(question, pdf_context, image_files):
    try:
        prompt_parts = [
            "You are a helpful assistant. Answer the question using the uploaded PDFs and images. "
            "If you don’t find the answer, say: \"I couldn't find that in the provided files.\
                1. First, look for relevant information in the PDFs. "
            "2. If needed, analyze the images for additional context.",
            "3. If you still can't find the answer, respond with: \"I couldn't find that in the provided files.\"",
            "4. If the question is not related to the uploaded files, respond with: \"I can't answer that question as it is not related to the uploaded files.\"",
            "5. If the question is about the content of the PDFs, provide a detailed answer based on the PDF context.",
            "6. If the question is about the images, provide a detailed answer based on the image context.",
            "7. If the question is about both PDFs and images, combine the information from both sources.",
            "8. If the question is about a specific topic, provide a detailed answer based on the context of the PDFs and images.",
            "9. If the question is about a specific file, provide a detailed answer based on the content of that file.",
            "10. If the question is about a specific image, provide a detailed answer based on  the content of that image.",
            "11. Analyse the pdf and images completely before answering the question.",
            "12. Provide a concise answers. ",
            "13. If the question is about a specific topic, provide a detailed answer based on the context of the PDFs and images.",
            "14. Stick to the context of the image content first before going to the pdf content.",
            "15. Stick to the current question before searching the json file.",
            "16. Provide the answer in point format. Give the heading to each point and then explain"
            f"PDF Context:\n{pdf_context[:8000]}",
            f"Question:\n{question}"
        ]
        # Add images to the prompt if any
        for img_file in image_files:
            img = Image.open(img_file)
            prompt_parts.append(img)
        # Use vision model for Q&A
        response = vision_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Display uploads
if uploaded_pdfs or uploaded_images:
    st.subheader("📄 Uploaded")

# PDF handling
# pdf_text = ""
# if uploaded_pdfs:
#     st.success(f"✅ {len(uploaded_pdfs)} PDF(s) uploaded successfully!")
#     pdf_text = extract_text_from_pdfs(uploaded_pdfs)
#     with st.spinner("🔍 Indexing PDFs into Vector Database..."):
#         store_chunks_and_embeddings(pdf_text)
#     st.success("✅ PDF content indexed in vector DB.")

# --- PDF handling ---
# --- PDF handling ---
# pdf_text = ""
# uploaded_pdf_info = [(f.name, f.size) for f in uploaded_pdfs] if uploaded_pdfs else []

# if uploaded_pdfs:
#     st.success(f"✅ {len(uploaded_pdfs)} PDF(s) uploaded successfully!")
#     pdf_text = extract_text_from_pdfs(uploaded_pdfs)

#     # Always save to vector DB if PDFs have changed
#     if (
#         "last_uploaded_pdfs" not in st.session_state
#         or st.session_state.last_uploaded_pdfs != uploaded_pdf_info
#         or st.session_state.faiss_index is None
#     ):
#         with st.spinner("🔍 Indexing PDFs into Vector Database..."):
#             store_chunks_and_embeddings(pdf_text)
#         st.session_state.last_uploaded_pdfs = uploaded_pdf_info
#         st.success("✅ PDF content indexed in vector DB.")
#     else:
#         st.info("ℹ️ Using existing vector DB (no new PDFs uploaded).")

# --- PDF handling ---
if "all_uploaded_pdfs" not in st.session_state:
    st.session_state.all_uploaded_pdfs = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if uploaded_pdfs:
    for f in uploaded_pdfs:
        if (f.name, f.size) not in [(pf.name, pf.size) for pf in st.session_state.all_uploaded_pdfs]:
            st.session_state.all_uploaded_pdfs.append(f)
    st.success(f"✅ {len(st.session_state.all_uploaded_pdfs)} PDF(s) in knowledge base!")
    # Extract text only for new PDFs
    st.session_state.pdf_text = extract_text_from_pdfs(st.session_state.all_uploaded_pdfs)

# --- Image handling ---
if "last_uploaded_images" not in st.session_state:
    st.session_state.last_uploaded_images = []

if "image_text" not in st.session_state:
    st.session_state.image_text = ""

current_uploaded_images = [(img.name, img.size) for img in uploaded_images] if uploaded_images else []

if uploaded_images and st.session_state.last_uploaded_images != current_uploaded_images:
    st.success(f"✅ {len(uploaded_images)} image(s) uploaded successfully!")
    with st.spinner("🔍 Extracting image understanding using Gemini..."):
        st.session_state.image_text = extract_insight_from_images(uploaded_images)
    st.session_state.last_uploaded_images = current_uploaded_images
    st.info("✅ Image content successfully processed.")
elif uploaded_images:
    st.info("ℹ️ Using existing image understanding (no new images uploaded).")

# --- Combine for vector DB ---
combined_text = st.session_state.pdf_text
if st.session_state.image_text:
    combined_text += "\n" + st.session_state.image_text

uploaded_pdf_info = [(f.name, f.size) for f in st.session_state.all_uploaded_pdfs]
if (
    (uploaded_pdfs or uploaded_images) and
    ("last_uploaded_pdfs" not in st.session_state or st.session_state.last_uploaded_pdfs != uploaded_pdf_info or st.session_state.faiss_index is None)
):
    with st.spinner("🔍 Indexing PDFs and images into Vector Database..."):
        store_chunks_and_embeddings(combined_text)
    st.session_state.last_uploaded_pdfs = uploaded_pdf_info
    st.success("✅ PDF and image content indexed in vector DB.")
elif uploaded_pdfs or uploaded_images:
    st.info("ℹ️ Using existing vector DB (no new PDFs/images uploaded).")

# Image handling
# image_text = ""
# if uploaded_images:
#     st.success(f"✅ {len(uploaded_images)} image(s) uploaded successfully!")
#     # for img_file in uploaded_images:
#     #     img = Image.open(img_file)
#     #     st.image(img, caption=f"{img_file.name}", use_column_width=True)
#     with st.spinner("🔍 Extracting image understanding using Gemini..."):
#         image_text = extract_insight_from_images(uploaded_images)
#     st.info("✅ Image content successfully processed.")
#     # st.write("🔎 **Extracted image context:**")
#     # st.write(image_text)

# Warning if nothing uploaded
if not uploaded_pdfs and not uploaded_images:
    st.warning("📎 Please upload at least a PDF or an image to begin.")

# Chat interface
# if pdf_text or uploaded_images:
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_input = st.chat_input("Ask a question based on the uploaded PDFs and images...")
#     if user_input:
#         st.chat_message("user").markdown(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         # Load previous questions
#         questions = load_questions()
#         # cached_answer = find_similar_question(user_input, questions)

#         # if cached_answer:
#         #     answer = f"🔁 (From previous answer)\n\n{cached_answer}"
#         # else:
#         with st.spinner("🤖 Thinking..."):
#             answer = ask_combined_question(user_input, pdf_text, uploaded_images)
#         # Save new question, answer, and embedding
#         questions.append({
#             "question": user_input,
#             # "embedding": embed_text(user_input),
#             "answer": answer
#         })
#         save_questions(questions)

#         st.chat_message("assistant").markdown(answer)
#         st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.pdf_text or st.session_state.image_text:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question based on the uploaded PDFs and images...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        questions = load_questions()
        # cached_answer = find_similar_question(user_input, questions)

        with st.spinner("🤖 Thinking..."):
            answer = ask_combined_question(
                user_input,
                st.session_state.pdf_text,
                uploaded_images
            )
        questions.append({
            "question": user_input,
            "answer": answer
        })
        save_questions(questions)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})