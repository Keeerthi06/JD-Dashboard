# JD-Dashboard

!pip install langchain langchain-community langchainhub langchain-openai langchain-chroma bs4 google-generativeai langchain-google-genai sentence-transformers==2.2.2 PyPDF2 python-docx

import os
import langchain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.colab import files

API_KEY = 'AIzaSyCdC68JCX4Om-GF7LCCcsey-PAbXPyQO9g'
os.environ['GOOGLE_API_KEY'] = API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-pro")

import re
import docx
import gradio as gr

# Define sections and keywords
sections_map = {
    "Technical Skills": ["technical skills", "technologies", "skills"],
    "Soft Skills": ["soft skills", "communication", "interpersonal"],
    "Experience": ["experience", "years of experience"],
    "Qualifications": ["qualifications", "education", "degree"],
    "Responsibilities": ["responsibilities", "duties", "role"],
    "Location": ["location", "work environment"],
    "Cloud Services": ["cloud services", "infrastructure"]
}

# Extract text from a .docx file
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text

# Clean JD text
def clean_text(text):
    text = re.sub(r'(Page \d+ of \d+|Manager/Supervisor.*Date|Department Manager.*Date|CERTIFICATION.*)', '', text, flags=re.DOTALL)
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# Extract sections from cleaned JD text
def extract_sections(text):
    sections = {key: [] for key in sections_map}
    current_section = None

    for line in text.splitlines():
        for section, keys in sections_map.items():
            if any(key.lower() in line.lower() for key in keys):
                current_section = section
                break
        if current_section and line.strip():
            sections[current_section].append(line.strip())
    return sections

# Generate markdown dashboards for each section
def create_dashboard(sections):
    dashboard_outputs = []
    for title, lines in sections.items():
        content = "\n".join(lines) if lines else "No data found in this section."
        markdown = f"### {title}\n{content}\n\n---"
        dashboard_outputs.append(markdown)
    return "\n".join(dashboard_outputs)

# Main Gradio function
def process_docx(file):
    try:
        raw_text = extract_text_from_docx(file.name)
        cleaned_text = clean_text(raw_text)
        extracted = extract_sections(cleaned_text)
        return create_dashboard(extracted)
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

# Launch Gradio app
gr.Interface(
    fn=process_docx,
    inputs=gr.File(label="Upload Job Description (.docx)", file_types=[".docx"]),
    outputs=gr.Markdown(label="Extracted Job Information"),
    title="📄 JD Skill Extractor",
    description="Upload a .docx Job Description file to extract and display technical skills, responsibilities, location, etc.",
).launch()
