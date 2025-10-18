import streamlit as st
import os
from google import genai
from google.genai.types import Part
from io import BytesIO

# --- Configuration and Initialization ---

st.set_page_config(
    page_title="Gemini Document Diff Tool",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Use st.cache_resource to initialize the client only once per unique API key, 
# and manage its lifecycle to avoid the __del__ traceback.
@st.cache_resource(hash_funcs={genai.Client: lambda _: None})
def initialize_gemini_client(api_key):
    """
    Initializes and caches the Gemini client with the provided API key.
    The hash_funcs={} bypasses hashing the client object itself.
    """
    if not api_key:
        return None
    try:
        # Pass the key explicitly to the client instance
        return genai.Client(api_key=api_key)
    except Exception as e:
        # Catch API key validation errors or network issues during setup
        st.error(f"Error initializing client with provided key: {e}")
        return None


# --- Functions ---

def compare_documents(client_instance, file_a, file_b, prompt):
    """
    Handles file conversion and calls the Gemini API to compare two PDF documents.
    """
    try:
        # CRITICAL: Rewind file pointers to ensure full reading after Streamlit/prior reads
        file_a.seek(0)
        file_b.seek(0)
        
        # 1. Convert file data to Generative Part objects
        part_a = Part.from_bytes(
            data=file_a.read(),
            mime_type=file_a.type
        )
        
        part_b = Part.from_bytes(
            data=file_b.read(),
            mime_type=file_b.type
        )

        # 2. Construct the list of parts for the API call
        
        # ADDED: System Instruction to guide the model's output format
        system_instruction = (
            "You are an expert document comparison and analysis tool. "
            "Your response MUST be formatted using Markdown. If the user asks for differences, "
            "changes, or a summary of items, prioritize using a **Markdown table** for clarity. "
            "Use clear headings like 'Change Type', 'Original Content', and 'Revised Content'."
        )

        full_prompt = (
            f"Compare the 'Revised Document' (second PDF) against the 'Original Document' (first PDF). "
            f"Here is the specific analysis required: {prompt}"
        )
        
        contents = [
            "Original Document:", part_a,
            "Revised Document:", part_b,
            full_prompt
        ]
        
        # 3. Call the API, including the system instruction
        response = client_instance.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config={"system_instruction": system_instruction}
        )
        
        return response.text

    except Exception as e:
        # Catch API call errors (network, content safety, invalid file parts, etc.)
        return f"An error occurred during API call: {e}"


# --- Streamlit UI ---

def app():
    # 1. API Key Setup (Sidebar)
    
    # Check environment variable first
    api_key = os.environ.get("GEMINI_API_KEY")
    
    with st.sidebar:
        st.header("Configuration")
        
        if api_key:
            st.success("API Key found in environment variable (GEMINI_API_KEY).")
        else:
            # If not in env, ask the user for it
            st.warning("GEMINI_API_KEY environment variable not found. Please enter your key below.")
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter your API Key here",
                key="user_api_key"
            )
            api_key = api_key_input

    # 2. Initialize Client
    # Pass the key from environment or user input to the cached function
    client_instance = initialize_gemini_client(api_key)


    st.title("⚖️ Gemini Document Diff Tool")
    st.markdown(
        """
        Upload two versions of a document (Original and Revised). Gemini will compare them 
        and provide a detailed summary of the differences based on your specific prompt.
        """
    )
    
    st.divider()

    # 3. Main Document Uploaders
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file_a = st.file_uploader(
            "1. Upload Original Document (PDF)",
            type=['pdf'],
            key="file_a"
        )

    with col2:
        uploaded_file_b = st.file_uploader(
            "2. Upload Revised Document (PDF)",
            type=['pdf'],
            key="file_b"
        )
    
    # Prompt Text Area
    prompt = st.text_area(
        "3. Specify the Comparison Task",
        placeholder="e.g., 'List all section headings that were added or removed, using a table.' or 'Summarize the financial changes between the two reports using a table.'",
        height=100
    )

    # Submission Button
    if st.button("🚀 Compare Documents", type="primary"):
        # Initial checks
        if not client_instance:
            st.error("Cannot run analysis. Please provide a valid API Key in the sidebar configuration.")
            return

        if uploaded_file_a is None or uploaded_file_b is None or not prompt:
            st.warning("Please upload both PDF files and enter a comparison task before proceeding.")
            return

        # Core execution
        with st.spinner("Comparing documents... this analysis may take a moment."):
            # Run the comparison
            result_text = compare_documents(client_instance, uploaded_file_a, uploaded_file_b, prompt)
            
        # Display results
        st.subheader("Gemini Comparison Analysis")
        # Streamlit renders the returned Markdown string, including tables, correctly.
        st.markdown(result_text)

if __name__ == "__main__":
    app()
