import PyPDF2
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def create_embeddings(sentences):
    model = SentenceTransformer('msmarco-roberta-base-v3')

    embeddings = model.encode(sentences)

    return embeddings

def extract_sentences_from_pdf(file_path):
    sentences = []
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            sentences.extend(text.split('. '))
    return sentences

# Example usage
pdf_file_path = r"C:\Users\Thanos\PycharmProjects\chatbot\sample.pdf"
extracted_sentences = extract_sentences_from_pdf(pdf_file_path)


# Print the extracted sentences
# for sentence in extracted_sentences:
#     print(sentence)
