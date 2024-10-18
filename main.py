from docx import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from groq import Groq

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def create_vector_store(docx_file_path, embedding_function):
    """Create a vector store from a DOCX file."""
    text = extract_text_from_docx(docx_file_path)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    documents = text_splitter.split_text(text)

    # Create Chroma vector store
    db = Chroma.from_texts(documents, embedding_function)

    return db

def get_bot_response(user_input, db, embedding_function, groq_client):
    """Get the bot response based on user input."""
    # Generate embedding for the user query
    query_embedding = embedding_function.embed_query(user_input)

    # Perform similarity search using Chroma
    similar_chunks = db.similarity_search_by_vector(query_embedding)

    # Gather context from similar chunks
    context = " ".join(chunk.page_content for chunk in similar_chunks)

    # Construct the detailed prompt
    detailed_prompt = f"You are a question-answering chatbot. Answer the following question: {user_input} \nContext: {context}"

    # Make a call to Groq API for chat completions
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": detailed_prompt}
        ],
        model="llama3-8b-8192",
        max_tokens=1000
    )

    return chat_completion.choices[0].message.content

def main():
    """Main function to run the chatbot."""
    # Get user inputs
    user_input = input("ENTER_YOUR_QUERY_HERE: ")
    API_KEY = input("ENTER_YOUR_GROQ_API_KEY_HERE: ")
    docx_file_path = input("ENTER_FILE_PATH: ")

    # Initialize Groq client
    groq_client = Groq(api_key=API_KEY)

    # Initialize embedding function
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store
    db = create_vector_store(docx_file_path, embedding_function)

    # Get and print the bot response
    response = get_bot_response(user_input, db, embedding_function, groq_client)
    print(response)

if __name__ == "__main__":
    main()
