import requests
from pdfminer.high_level import extract_text
import streamlit as st
import os
from dotenv import load_dotenv # Import load_dotenv

# --- Configuration ---
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Load environment variables from .env file (for local development)
# This should be called before trying to access os.getenv()
load_dotenv() 

# IMPORTANT: Securely get your API Key.
# It's highly recommended to use environment variables for API keys.
# Ensure your .env file in the project root has a line like: NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
API_KEY = os.getenv("NVIDIA_API_KEY") 

# --- Helper Functions ---

def extract_resume_text(pdf_file):
    """
    Extracts text content from an uploaded PDF file.
    Handles potential errors during text extraction.
    """
    try:
        # Read the file-like object directly
        text = extract_text(pdf_file)
        if not text.strip(): # Check if text is empty or just whitespace after extraction
            st.warning("⚠️ No readable text found in the PDF. It might be an image-only PDF. Consider converting it to a searchable PDF or try a different file.")
            return ""
        return text
    except Exception as e:
        st.error(f"🚫 **Error extracting text from PDF:** Please ensure it's a valid, readable PDF document. Details: {e}")
        return "" # Return empty string on error

def get_feedback_nvidia(resume_text, target_job_role=None):
    """
    Sends the resume text to the NVIDIA API's Llama 3 70B Instruct model
    and retrieves AI-generated feedback.
    """
    # Check if API_KEY is loaded correctly
    if not API_KEY or API_KEY == "nvapi-YOUR_ACTUAL_API_KEY_HERE" or API_KEY.startswith("nvapi-ziMpyqbV"):
        st.error("🚫 **API Key Error:** Your NVIDIA API Key is missing or invalid. Please ensure it's correctly set as an environment variable (e.g., in a `.env` file).")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Construct the system prompt based on whether a target job role is provided
    system_prompt_base = """You are a highly professional, empathetic, and constructive resume reviewer. Your goal is to help the user significantly improve their resume. Analyze the provided resume text thoroughly. Focus on the following aspects:

1.  **Overall Impact:** Does it make a strong first impression?
2.  **Clarity & Conciseness:** Is the language clear, and is it free of jargon or fluff?
3.  **Action Verbs:** Are strong action verbs used at the beginning of bullet points?
4.  **Quantifiable Achievements:** Are accomplishments quantified with numbers/data where possible?
5.  **Keywords & Industry Relevance:** Does it include relevant keywords for the target industry/role?
6.  **Structure & Formatting:** Is it well-organized and easy to read (e.g., consistent dates, clear sections)?
7.  **Tailoring:** Does it seem tailored to a specific job or industry?

Provide actionable feedback with specific examples. Suggest improvements for content, wording, and potential areas to expand or condense. Structure your feedback into the following distinct sections, using Markdown for clear formatting (e.g., bullet points, bolding, subheadings):

### Overall Summary
### Strengths 💪
### Areas for Improvement 💡
### Actionable Steps 🚀
"""
    if target_job_role and target_job_role.strip(): # Check if target_job_role is provided and not just whitespace
        system_prompt_base += f"\n\n**Special consideration:** The user is targeting a **{target_job_role.strip()}** role. Please tailor your keyword and relevance analysis to this role."

    payload = {
        "model": "meta/llama3-70b-instruct", # The exact model identifier
        "messages": [
            {"role": "system", "content": system_prompt_base},
            {"role": "user", "content": f"Here is the resume text to review:\n\n{resume_text}"}
        ],
        "max_tokens": 1500, # Max tokens for comprehensive feedback
        "temperature": 0.5, # Balance creativity and consistency
        "top_p": 0.9 # For diverse but coherent responses
    }

    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        if "choices" in response_json and response_json["choices"]:
            first_choice = response_json["choices"][0]
            if "message" in first_choice and "content" in first_choice["message"]:
                return first_choice["message"]["content"]
        
        st.warning("⚠️ Received an empty or unexpected response format from the AI. Please try again.")
        return None

    except requests.exceptions.RequestException as e:
        # Catch and report any request-related errors (e.g., network issues, invalid URL, authentication errors)
        st.error(f"🚫 **Network or API Error:** {e}. Please ensure your internet connection is stable and your API key is valid and has sufficient credits/access.")
        return None
    except Exception as e:
        # Catch any other unexpected errors during response parsing
        st.error(f"❌ An unexpected error occurred while processing the AI response: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(
    page_title="🤖 AI Resume Reviewer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for a slightly better look
st.markdown("""
<style>
.stApp {
    background-color: #0d1117; /* Dark background */
    color: #e6edf3; /* Light text */
}
.reportview-container .main .block-container{
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 1rem;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.1rem;
}
h1 {
    color: #4CAF50; /* A nice green for the main title */
    text-align: center;
}
h2, h3, h4 {
    color: #8f9ba8; /* Slightly lighter for subheadings */
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stTextInput>div>div>input {
    border: 1px solid #30363d;
    border-radius: 5px;
    background-color: #0d1117;
    color: #e6edf3;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Reviewer")
st.markdown("### Get Instant, AI-Powered Feedback on Your Resume!")
st.markdown("""
Upload your resume (PDF) below, and our intelligent AI will analyze it for clarity, impact, keywords, and more. 
Get actionable suggestions to make your resume stand out to recruiters!
""")

# --- User Input Section ---
st.header("1️⃣ Upload Your Resume")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Please upload your resume in PDF format.")

st.header("2️⃣ Tell Us About Your Target Job (Optional)")
target_role = st.text_input(
    "Enter the job title you're applying for (e.g., 'Software Engineer', 'Marketing Manager')",
    placeholder="e.g., Data Scientist, Project Manager",
    help="Providing a target job role helps the AI tailor its feedback, especially for keywords and relevance."
)

st.markdown("---")

# --- Feedback Generation Trigger ---
if uploaded_file is not None:
    # Use a container for the button and spinner to ensure consistent layout
    button_col, _ = st.columns([0.4, 0.6])
    with button_col:
        if st.button("✨ Get Resume Feedback!", type="primary"):
            with st.spinner("🚀 Analyzing your resume and generating personalized feedback... This might take up to 30-60 seconds for detailed analysis."):
                resume_text = extract_resume_text(uploaded_file)

                if resume_text: # Only proceed if text extraction was successful
                    # Corrected variable name: using 'target_role' as defined by st.text_input
                    feedback = get_feedback_nvidia(resume_text, target_role) 
                    if feedback:
                        st.header("📝 Your AI-Powered Resume Feedback:")
                        st.success("🎉 Feedback Generated Successfully!")
                        st.markdown(feedback) # Render AI feedback with Markdown formatting
                        st.info("💡 **Remember:** This is AI-generated feedback. Always review it critically and tailor your resume to specific job applications. Human review is always recommended!")
                    # Error messages are now handled inside get_feedback_nvidia
                # else: extract_resume_text already shows an error if text is empty

# --- About Section ---
st.markdown("---")
with st.expander("❓ About This AI Resume Reviewer"):
    st.markdown("""
    This application utilizes the powerful **Llama 3 70B Instruct** model via the NVIDIA AI Foundation Models API. 
    It's designed to provide instant, constructive feedback on your resume based on best practices for job applications.
    
    **Features:**
    -   PDF text extraction
    -   AI analysis for impact, clarity, action verbs, quantification, keywords, and structure.
    -   Tailored feedback based on an optional target job role.
    
    **Disclaimer:** While powerful, AI feedback is not a substitute for professional career advice. Always use your best judgment.
    """)

st.markdown("---")
st.caption("© 2025 AI Resume Reviewer. Powered by Streamlit & NVIDIA AI Foundation Models.")