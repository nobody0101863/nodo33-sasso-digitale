#!/usr/bin/env python3
"""
RAG Locale - Retrieval Augmented Generation
Carica PDF, libri, documenti → Chiedi qualsiasi cosa
ZERO API esterne. Tutto locale.
"""

import argparse
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGLocale:
    def __init__(self, model="llama3.1:70b", persist_directory="./chroma_db"):
        """Inizializza il sistema RAG con Ollama locale"""
        self.model = model
        self.persist_directory = persist_directory

        # Embeddings locali (per vettorizzare il testo)
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # modello di embedding locale
            base_url="http://localhost:11434"
        )

        # LLM locale
        self.llm = Ollama(
            model=self.model,
            base_url="http://localhost:11434"
        )

        # Vector store (database vettoriale)
        self.vectorstore = None
        self.qa_chain = None

    def carica_pdf(self, pdf_path: str):
        """Carica un PDF e lo inserisce nel database vettoriale"""
        print(f"📖 Caricamento PDF: {pdf_path}")

        # Carica il PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"   Trovate {len(documents)} pagine")

        # Dividi in chunk (pezzi gestibili)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        print(f"   Creati {len(splits)} chunks di testo")

        # Crea il vector store
        print("   Creazione embeddings (può richiedere qualche minuto)...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print("✅ PDF caricato con successo!")

    def carica_directory(self, dir_path: str):
        """Carica tutti i PDF in una directory"""
        pdf_files = list(Path(dir_path).glob("*.pdf"))
        print(f"📚 Trovati {len(pdf_files)} PDF in {dir_path}")

        for pdf_file in pdf_files:
            self.carica_pdf(str(pdf_file))

    def crea_qa_chain(self):
        """Crea la catena di question-answering"""
        if not self.vectorstore:
            print("❌ Errore: Nessun documento caricato!")
            return

        # Template del prompt (personalizzabile)
        prompt_template = """Usa il seguente contesto per rispondere alla domanda.
Se non sai la risposta, dì semplicemente "Non lo so", non inventare.

Contesto: {context}

Domanda: {question}

Risposta dettagliata e accurata:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Crea la chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        print("✅ Sistema RAG pronto per le domande!")

    def chiedi(self, query: str):
        """Fai una domanda al sistema RAG"""
        if not self.qa_chain:
            self.crea_qa_chain()

        print(f"\n🔍 Domanda: {query}")
        print("Cerco nei documenti...\n")

        result = self.qa_chain.invoke({"query": query})

        print("=" * 70)
        print("🎯 RISPOSTA:")
        print("=" * 70)
        print(result["result"])
        print("\n" + "=" * 70)
        print("📄 FONTI:")
        print("=" * 70)

        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "Sconosciuto")
            page = doc.metadata.get("page", "?")
            print(f"\n{i}. {source} (pagina {page + 1})")
            print(f"   {doc.page_content[:200]}...")

def main():
    parser = argparse.ArgumentParser(
        description="RAG Locale - Interroga i tuoi PDF con AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Interroga un singolo PDF
  python rag_locale.py --pdf "De_Rerum_Natura.pdf" --query "Cosa dice dell'atomo?"

  # Carica tutti i PDF in una directory
  python rag_locale.py --dir "./libri" --query "Cos'è la filosofia stoica?"

  # Usa un modello diverso
  python rag_locale.py --pdf "paper.pdf" --model "mistral-large" --query "Riassumi il paper"
        """
    )

    parser.add_argument("--pdf", help="Path al PDF da interrogare")
    parser.add_argument("--dir", help="Directory contenente PDF")
    parser.add_argument("--query", required=True, help="La tua domanda")
    parser.add_argument("--model", default="llama3.1:70b", help="Modello Ollama da usare")
    parser.add_argument("--db", default="./chroma_db", help="Directory per il database vettoriale")

    args = parser.parse_args()

    if not args.pdf and not args.dir:
        parser.error("Specifica --pdf o --dir")

    # Banner
    print("🪨❤️ NODO33 - RAG Locale (Sapienza Gratis)")
    print("=" * 70)

    # Verifica Ollama
    try:
        rag = RAGLocale(model=args.model, persist_directory=args.db)
    except Exception as e:
        print("❌ Errore: Ollama non è attivo!")
        print("\nAvvia Ollama:")
        print("  ollama serve")
        print("\nScarica i modelli necessari:")
        print("  ollama pull llama3.1:70b")
        print("  ollama pull nomic-embed-text")
        return 1

    # Carica documenti
    if args.pdf:
        rag.carica_pdf(args.pdf)
    elif args.dir:
        rag.carica_directory(args.dir)

    # Fai la domanda
    rag.chiedi(args.query)

    print("\n🪨 'Se anche costoro taceranno, grideranno le pietre!'")

if __name__ == "__main__":
    main()
