import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi
from openai import OpenAI
from PyPDF2 import PdfReader
import bs4
from langchain_community.document_loaders import WebBaseLoader
from tqdm import tqdm
from typing import List, Dict, Tuple
import pickle
from chat_session import ChatSession

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RAGSystem:

    def __init__(self, openai_api_key: str, openai_base_url: str, hf_api_key: str, chunk_size: int = 768, index_directory: str = "saved_index"):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        # Initialize chat session
        self.chat_session = ChatSession()

        # Initialize Hugging Face API
        self.hf_api = HfApi(token=hf_api_key)
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize FAISS index
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.chunk_size = chunk_size
        self.text_chunks = []
        self.metadata = []
        self.index_directory = index_directory

        # Intentar cargar índice existente o crear uno nuevo
        self.initialize_or_load_index()

    def initialize_or_load_index(self):
        """Intenta cargar el índice existente, si no existe, crea uno nuevo."""
        try:
            print("Intentando cargar índice existente...")
            self.load_index(self.index_directory)
            print(f"Índice cargado exitosamente con {len(self.text_chunks)} chunks de texto")
        except (FileNotFoundError, Exception) as e:
            print("No se encontró índice existente o hubo un error al cargarlo.")
            print("Creando nuevo índice a partir de los documentos...")
            self.load_all_documents()
            print("Guardando nuevo índice...")
            self.save_index(self.index_directory)
            print(f"Nuevo índice creado y guardado con {len(self.text_chunks)} chunks de texto")

    def load_all_documents(self):
        """Carga todos los documentos, tanto PDFs como páginas web."""
        print("Procesando documentos...")
        # Cargar PDFs
        self.load_documents_from_pdfs()
        # Cargar páginas web
        self.load_documents_from_web()
        print("Documentos procesados y embeddings generados.")

    def load_documents_from_pdfs(self):
        """Carga todos los documentos de la carpeta pdfs."""
        pdfs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
        if not os.path.exists(pdfs_dir):
            print(f"Advertencia: La carpeta {pdfs_dir} no existe.")
            return

        print("Cargando documentos de la carpeta pdfs...")
        for filename in os.listdir(pdfs_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdfs_dir, filename)
                try:
                    self.process_pdf(pdf_path)
                    print(f"Documento cargado exitosamente: {filename}")
                except Exception as e:
                    print(f"Error al cargar {filename}: {str(e)}")

    def load_documents_from_web(self):
        """Carga documentos desde páginas web específicas."""
        print("Cargando documentos web...")
        web_paths = [
            "https://www.churchofjesuschrist.org/study/manual/general-handbook/10-aaronic-priesthood?lang=spa#title_number22",
            "https://www.churchofjesuschrist.org/study/manual/general-handbook/6-stake-leadership?lang=spa#title_number2",
            "https://www.churchofjesuschrist.org/study/manual/general-handbook/8-elders-quorum?lang=spa#title_number2",
            "https://www.churchofjesuschrist.org/study/manual/general-handbook?lang=spa",
            "https://www.churchofjesuschrist.org/study/manual/general-handbook/summary-of-recent-updates?lang=spa",
            "https://www.churchofjesuschrist.org/study/liahona/2021/12/digital-only/5-ways-to-use-the-light-the-world-calendar-at-home?lang=spa",
        ]
        
        bs4_strainer = bs4.SoupStrainer(class_=("contentWrapper-n6Z8K"))
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        
        try:
            docs = loader.load()
            for doc in docs:
                # Procesar el contenido del documento web
                text = doc.page_content
                chunks = self._split_text(text)
                
                # Crear embeddings y agregar al índice
                embeddings = self._create_embeddings(chunks)
                self.index.add(embeddings)
                
                # Guardar chunks y metadata
                start_idx = len(self.text_chunks)
                self.text_chunks.extend(chunks)
                
                # Agregar metadata para cada chunk
                for i, chunk in enumerate(chunks):
                    self.metadata.append({
                        'source': doc.metadata.get('source', 'web'),
                        'chunk_index': start_idx + i
                    })
                
            print(f"Se cargaron {len(docs)} documentos web exitosamente")
        except Exception as e:
            print(f"Error al cargar documentos web: {str(e)}")

    def _split_text(self, text: str) -> List[str]:
        """Divide el texto en chunks del tamaño especificado."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > self.chunk_size:
                if current_chunk:  # only append if we have something
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:  # append the last chunk if it exists
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_pdf(self, pdf_path: str):
        """Procesa un archivo PDF, extrae el texto y lo divide en chunks."""
        print(f"Procesando PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Dividir el texto en chunks
        chunks = self._split_text(text)
        
        # Crear embeddings y agregar al índice
        embeddings = self._create_embeddings(chunks)
        self.index.add(embeddings)
        
        # Guardar chunks y metadata
        start_idx = len(self.text_chunks)
        self.text_chunks.extend(chunks)
        
        # Agregar metadata para cada chunk
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                'source': os.path.basename(pdf_path),
                'type': 'pdf',
                'chunk_index': start_idx + i
            })

        print(f"PDF procesado: {len(chunks)} chunks creados")

    def process_directory(self, directory_path: str) -> None:
        """Process all PDF files in a directory."""
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                self.process_pdf(pdf_path)

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of text chunks."""
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Creating embeddings"):
                # Tokenize and encode text
                inputs = self.tokenizer(text, padding=True, truncation=True, 
                                     max_length=512, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

    def query(self, query_text: str, k: int = 3) -> str:
        """
        Realiza una consulta al sistema RAG.
        
        Args:
            query_text (str): Texto de la consulta
            k (int): Número de documentos similares a recuperar
            
        Returns:
            str: Respuesta generada
        """
        try:
            # Generar embedding para la consulta
            query_embedding = self._create_embeddings([query_text])
            
            # Buscar documentos similares
            D, I = self.index.search(query_embedding, k)
            
            # Obtener los textos relevantes
            relevant_texts = [self.text_chunks[i] for i in I[0]]
            sources = [self.metadata[i] for i in I[0]]
            
            # Construir el contexto con los textos relevantes
            context = "\n\n".join(relevant_texts)
            
            # Construir el prompt
            messages = [
                {"role": "system", "content": f"Eres un asistente SUD  (nunca decir iglesia mormona). Usa el siguiente contexto para responder la pregunta del usuario:\n\nContexto:\n{context}"},
                {"role": "user", "content": query_text}
            ]
            
            # Obtener respuesta de OpenAI
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.01,
                max_tokens=1500
            )
            
            # Construir respuesta con fuentes
            answer = response.choices[0].message.content
            
            # Agregar información sobre las fuentes consultadas
            sources_info = "\n\nFuentes consultadas:"
            for i, (text, source) in enumerate(zip(relevant_texts, sources), 1):
                sources_info += f"\n{i}. {source.get('source', 'Desconocido')}"
                sources_info += f"\nFragmento: {text[:150]}..."  # Mostrar los primeros 150 caracteres
            
            final_response = answer + sources_info
            
            # Actualizar historial de chat
            self.chat_session.add_message("user", query_text)
            self.chat_session.add_message("assistant", final_response)
            
            return final_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Error al procesar la pregunta: {str(e)}")

    def save_index(self, directory: str = "saved_index"):
        """
        Guarda el índice FAISS y los datos relacionados en un directorio.
        
        Args:
            directory (str): Directorio donde se guardarán los archivos
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar el índice FAISS
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Guardar los chunks de texto y metadata
        with open(os.path.join(directory, "text_chunks.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)
        
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self, directory: str = "saved_index"):
        """
        Carga el índice FAISS y los datos relacionados desde un directorio.
        
        Args:
            directory (str): Directorio desde donde se cargarán los archivos
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"El directorio {directory} no existe")
            
        # Cargar el índice FAISS
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Cargar los chunks de texto
        chunks_path = os.path.join(directory, "text_chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                self.text_chunks = pickle.load(f)
        
        # Cargar metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def clear_chat_history(self):
        """Clear the chat session history."""
        self.chat_session.clear_history()