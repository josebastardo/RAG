import streamlit as st
from rag_system import RAGSystem
from config import Config
from dotenv import  load_dotenv

load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Chat python SUD",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f0f2f6;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize RAG system
@st.cache_resource
def init_rag():
    try:
        Config.validate_config()
        rag = RAGSystem(
            hf_api_key=Config.HF_API_KEY,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_base_url=Config.OPENAI_BASE_URL,
        )
        return rag
    except Exception as e:
        st.error(f"Error al inicializar el sistema: {str(e)}")
        return None

def main():
    st.title("📚 Chat con Documentos SUD")
    st.header("🌟 Ilumina el Mundo")
    st.markdown("""
        La Barrio Sucre te invita a ser parte de la iniciativa 'Ilumina el Mundo'.
        Siguiendo el ejemplo de Jesucristo, podemos compartir Su luz a través del servicio y de amor.
        Visita [IluminaElMundo.org](https://www.iluminaelmundo.org) para más recursos.
        """)
    

    st.markdown("""
        ### ¡Bienvenido a ChatySUD!
        Este es un espacio no oficial para consultar y aprender sobre documentos SUD.
        Puedes hacer preguntas sobre los documentos del manual azul y mantener una conversación conmigo.
                
     💡 Recuerda:  Para decisiones importantes, consulta siempre con tus líderes locales.
        """)
    
        # Initialize RAG system
    rag = init_rag()
    if not rag:
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Hazme una pregunta sobre los documentos..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = rag.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error detallado: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
    
    # Sidebar
    with st.sidebar:
        st.header("📖 Acerca de")
        st.markdown("""
        Este chat fue creado para ayudarte a encontrar respuestas
        en los documentos oficiales de la Iglesia de Jesucristo de los Santos de los Últimos Días.
        """)

        st.markdown("---")

        
        # Clear chat
        if st.button("🗑️ Limpiar Chat", help="Elimina el historial de la conversación"):
            st.session_state.messages = []
            rag.clear_chat_history()
            st.rerun()


        st.markdown("---")
        st.header("🌟 Ilumina el Mundo")
        st.markdown("""
        #### ¿Cómo puedes participar?
        * Sirve a tu prójimo siguiendo el ejemplo de Jesucristo
        * Invita a otros a conocer más sobre el evangelio
        * Participa en actividades de servicio en tu comunidad
        
        Visita [IluminaElMundo.org](https://www.iluminaelmundo.org) para más información y recursos.
        """)    

        st.markdown("---")
        
        st.header("👨‍👩‍👧‍👦 FamilySearch")
        st.markdown("""
        ### Descubre Tu Historia Familiar
        
        FamilySearch es el recurso genealógico más grande del mundo, ¡y es completamente gratuito!
        
        #### Lo que puedes hacer:
        * Crear tu árbol genealógico
        * Buscar registros históricos
        * Preservar fotos y memorias familiares
        * Conectar con familiares
        * Encontrar antepasados para la obra del templo
        
        
        [Visita FamilySearch.org](https://www.familysearch.org/es/) para comenzar tu búsqueda familiar.
        """)

    # Espacio para asegurar que el footer aparezca después del chat
    st.markdown("<br><br>", unsafe_allow_html=True)
        
    

if __name__ == "__main__":
    main()