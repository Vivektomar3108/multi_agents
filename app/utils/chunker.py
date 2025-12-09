from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,                   # sweet spot
        chunk_overlap=150,                 # not too large
        length_function=len,
        
        # BEST separator order (from strongest to weakest)
        separators=[
            "\n\n",                        # paragraph break
            "\n",                          # line break
            ". ",                          # sentence break
            " ",                           # word break
            ""                             # fallback to characters
        ],
    )
