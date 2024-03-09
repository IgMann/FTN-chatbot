# GENERAL CONFIGURATION

# Paths
DATABASE_PATH = "vectorstore/db_faiss"

# Context Embedding and base creation
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cpu"

#Chatbot parameters
REPLICATE_API_TOKEN = "your replicate token"
# CHATBOT_MODEL = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
CHATBOT_MODEL = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
TEMPERATURE = 0.01
TOP_PERCENT = 0.5
MAX_NEW_TOKENS = 1000

# Chat parameters

# Optional initial history
CHAT_HISTORY = [
    # ("U kom gradu se nalazi Fakultet tehničkih nauka?", "U Novom Sadu."),
    # ("Kad je osnovan?", "1960."),
    # ("Koja je web stranica fakulteta?", "www.ftn.uns.ac.rs")
]

FAITHFULNESS_LIMIT = 3
DEFAULT_ANSWER = """
    Na žalost, ne mogu da odgovorim sa sigurnošću na vaše pitanje.
    Za više informacija posetite sajt fakulteta "www.ftn.uns.ac.rs" ili se obratite studentskoj službi lično.
    """
