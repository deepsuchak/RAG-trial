import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from llama_index.readers.file import PDFReader

loader = PDFReader()

def get_pdf_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        index = VectorStoreIndex.from_documents(data,show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir=index_name))
        

    return index


pdf_path = os.path.join("data","data-space.pdf")
space_pdf = PDFReader().load_data(pdf_path)
space_index  = get_pdf_index(space_pdf,"data-space")
space_engine = space_index.as_query_engine()
