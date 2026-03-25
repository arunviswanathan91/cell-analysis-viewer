"""
TRUE RAG SYSTEM - Answers ANY question about your data

This is REAL Retrieval-Augmented Generation:
1. Embeds ALL data into vectors (signatures, analysis, cell types, interactions)
2. Semantic search retrieves relevant chunks for ANY query
3. LLM generates answer based on retrieved context

NO hardcoding. NO anticipated queries. Just semantic search + generation.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Vector store
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence transformers not installed. Run: pip install sentence-transformers")

# LLM for generation
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.vocabulary import VALID_CELL_TYPES, SYSTEM_PROMPT_REWRITER

from dotenv import load_dotenv
load_dotenv()


class TrueRAG:
    """
    True RAG system that can answer ANY question about your data.

    No hardcoded queries. No rule-based routing.
    Just: Embed -> Retrieve -> Generate
    """

    def __init__(self, data_dir: str = None, persist_dir: str = None):
        self.base_path = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.base_path / "data2"
        self.persist_dir = Path(persist_dir) if persist_dir else self.base_path / "vector_store"

        # Initialize embedding model (runs locally, free)
        if EMBEDDINGS_AVAILABLE:
            print("Loading embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
        else:
            self.embedder = None

        # Initialize vector store
        if CHROMA_AVAILABLE:
            self.persist_dir.mkdir(exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.chroma_client.get_or_create_collection(
                name="cell_analysis_rag",
                metadata={"description": "Cell analysis data for RAG"}
            )
        else:
            self.chroma_client = None
            self.collection = None

        # Initialize LLM
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if GROQ_AVAILABLE else None

        # Track what's indexed
        self.index_hash_file = self.persist_dir / "index_hash.json"

    def _compute_data_hash(self) -> str:
        """Compute hash of data files to detect changes."""
        hasher = hashlib.md5()
        for f in sorted(self.data_dir.rglob("*.parquet")):
            hasher.update(f.name.encode())
            hasher.update(str(f.stat().st_mtime).encode())
        return hasher.hexdigest()

    def _needs_reindex(self) -> bool:
        """Check if we need to reindex."""
        if not self.index_hash_file.exists():
            return True
        try:
            with open(self.index_hash_file) as f:
                stored = json.load(f)
            return stored.get("hash") != self._compute_data_hash()
        except:
            return True

    def _save_index_hash(self):
        """Save current data hash."""
        with open(self.index_hash_file, 'w') as f:
            json.dump({"hash": self._compute_data_hash()}, f)

    def _load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all parquet files."""
        data = {}

        # Categorical results
        cat_path = self.data_dir / "categorical" / "results_all_comparisons.parquet"
        if cat_path.exists():
            data["categorical"] = pd.read_parquet(cat_path)

        # Continuous results
        cont_path = self.data_dir / "continuous" / "bmi_slope_results.parquet"
        if cont_path.exists():
            data["continuous"] = pd.read_parquet(cont_path)

        # Interactome
        inter_path = self.data_dir / "interactome" / "interactions_all_sources.parquet"
        if inter_path.exists():
            data["interactome"] = pd.read_parquet(inter_path)

        # Cell type mapping
        cell_path = self.data_dir / "reference" / "celltype_mapping_all.parquet"
        if cell_path.exists():
            data["cell_types"] = pd.read_parquet(cell_path)

        # Signatures
        sig_path = self.data_dir / "reference" / "signatures.parquet"
        if sig_path.exists():
            data["signatures"] = pd.read_parquet(sig_path)

        # Also load JSON signatures
        json_sig_path = self.base_path / "data" / "signatures" / "ALL_CELL_SIGNATURES_FLAT.json"
        if json_sig_path.exists():
            with open(json_sig_path) as f:
                data["signatures_json"] = json.load(f)

        return data

    def _create_documents(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Convert all data into text documents for embedding.
        Each document has: id, text, metadata
        """
        documents = []
        doc_id = 0

        # 1. CATEGORICAL ANALYSIS RESULTS
        if "categorical" in data:
            df = data["categorical"]
            for _, row in df.iterrows():
                cell_type = row.get("cell_type", "Unknown")
                feature = row.get("feature", "Unknown")

                # Create rich text description
                text_parts = [
                    f"Cell type: {cell_type}",
                    f"Signature/Feature: {feature}",
                ]

                # Add comparison results
                for comp in ["overweight_vs_normal", "obese_vs_normal", "obese_vs_overweight"]:
                    mean_col = f"{comp}_mean"
                    prob_col = f"{comp}_prob"
                    cred_col = f"{comp}_credible"

                    if mean_col in row:
                        mean_val = row[mean_col]
                        prob_val = row.get(prob_col, None)
                        is_cred = row.get(cred_col, False)

                        direction = "increased" if mean_val > 0 else "decreased"
                        comp_readable = comp.replace("_", " ").title()

                        text_parts.append(
                            f"{comp_readable}: {feature} is {direction} in {cell_type} "
                            f"(effect size: {mean_val:.3f}, probability: {prob_val:.2f if prob_val else 'N/A'}, "
                            f"credible: {is_cred})"
                        )

                doc_text = "\n".join(text_parts)
                documents.append({
                    "id": f"cat_{doc_id}",
                    "text": doc_text,
                    "metadata": {
                        "source": "categorical_analysis",
                        "cell_type": str(cell_type),
                        "feature": str(feature)
                    }
                })
                doc_id += 1

        # 2. CONTINUOUS BMI ANALYSIS
        if "continuous" in data:
            df = data["continuous"]
            for _, row in df.iterrows():
                cell_type = row.get("cell_type", "Unknown")
                feature = row.get("feature", "Unknown")
                slope = row.get("bmi_slope_mean", 0)

                direction = "increases" if slope > 0 else "decreases"

                text = (
                    f"BMI Dose-Response Analysis:\n"
                    f"Cell type: {cell_type}\n"
                    f"Signature: {feature}\n"
                    f"As BMI increases, {feature} {direction} in {cell_type} "
                    f"(slope: {slope:.4f})"
                )

                documents.append({
                    "id": f"cont_{doc_id}",
                    "text": text,
                    "metadata": {
                        "source": "continuous_analysis",
                        "cell_type": str(cell_type),
                        "feature": str(feature)
                    }
                })
                doc_id += 1

        # 3. INTERACTOME - CELL-CELL INTERACTIONS
        if "interactome" in data:
            df = data["interactome"]
            for _, row in df.iterrows():
                cell1 = row.get("favorable_cell", row.get("cell_type_1", "Unknown"))
                cell2 = row.get("unfavorable_cell", row.get("cell_type_2", "Unknown"))
                gene1 = row.get("gene1", row.get("ligand", "Unknown"))
                gene2 = row.get("gene2", row.get("receptor", "Unknown"))
                enrichment = row.get("enrichment_ratio", row.get("score", 0))

                text = (
                    f"Cell-Cell Interaction:\n"
                    f"Cells: {cell1} communicates with {cell2}\n"
                    f"Genes: {gene1} (ligand) -> {gene2} (receptor)\n"
                    f"Enrichment ratio: {enrichment:.2f}\n"
                    f"This interaction suggests {cell1} signals to {cell2} via {gene1}-{gene2} axis."
                )

                documents.append({
                    "id": f"inter_{doc_id}",
                    "text": text,
                    "metadata": {
                        "source": "interactome",
                        "cell_1": str(cell1),
                        "cell_2": str(cell2),
                        "gene_1": str(gene1),
                        "gene_2": str(gene2)
                    }
                })
                doc_id += 1

        # 4. CELL TYPE DEFINITIONS
        if "cell_types" in data:
            df = data["cell_types"]
            for _, row in df.iterrows():
                canonical = row.get("canonical_name", row.get("cell_type", "Unknown"))
                compartment = row.get("compartment", "Unknown")
                aliases = row.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]

                text = (
                    f"Cell Type Definition:\n"
                    f"Name: {canonical}\n"
                    f"Compartment: {compartment}\n"
                    f"Also known as: {', '.join(aliases) if aliases else 'N/A'}\n"
                    f"{canonical} is a {compartment} cell type in pancreatic cancer."
                )

                documents.append({
                    "id": f"cell_{doc_id}",
                    "text": text,
                    "metadata": {
                        "source": "cell_type_reference",
                        "cell_type": str(canonical),
                        "compartment": str(compartment)
                    }
                })
                doc_id += 1

        # 5. GENE SIGNATURES
        if "signatures_json" in data:
            for cell_type, signatures in data["signatures_json"].items():
                for sig_name, genes in signatures.items():
                    if isinstance(genes, list):
                        gene_list = genes[:20]  # First 20 genes
                        gene_str = ", ".join(gene_list)

                        text = (
                            f"Gene Signature:\n"
                            f"Signature name: {sig_name}\n"
                            f"Cell type: {cell_type}\n"
                            f"Number of genes: {len(genes)}\n"
                            f"Key genes: {gene_str}\n"
                            f"The {sig_name} signature in {cell_type} includes genes like {gene_str}."
                        )

                        documents.append({
                            "id": f"sig_{doc_id}",
                            "text": text,
                            "metadata": {
                                "source": "signature",
                                "signature_name": str(sig_name),
                                "cell_type": str(cell_type),
                                "gene_count": len(genes)
                            }
                        })
                        doc_id += 1

        print(f"Created {len(documents)} documents for embedding")
        return documents

    def index_data(self, force: bool = False):
        """
        Index all data into vector store.
        Only reindexes if data has changed (or force=True).
        """
        if not CHROMA_AVAILABLE or not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("ChromaDB and sentence-transformers required. Install them first.")

        if not force and not self._needs_reindex():
            print("Data already indexed and up to date.")
            return

        print("Indexing all data for RAG...")

        # Load all data
        data = self._load_all_data()

        # Create documents
        documents = self._create_documents(data)

        if not documents:
            print("No documents to index!")
            return

        # Clear existing collection
        try:
            self.chroma_client.delete_collection("cell_analysis_rag")
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name="cell_analysis_rag",
            metadata={"description": "Cell analysis data for RAG"}
        )

        # Batch embed and add
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            texts = [d["text"] for d in batch]
            ids = [d["id"] for d in batch]
            metadatas = [d["metadata"] for d in batch]

            # Embed
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            # Add to vector store
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            print(f"Indexed {min(i+batch_size, len(documents))}/{len(documents)} documents")

        self._save_index_hash()
        print(f"Indexing complete! {len(documents)} documents in vector store.")

    def rewrite_query(self, user_query: str) -> str:
        """
        Uses the LLM to translate natural language into database-specific terms.
        """
        if not self.groq_client:
            return user_query

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_REWRITER},
            {"role": "user", "content": f"User Query: {user_query}\nTarget Search Terms:"}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Stronger model for reasoning
                messages=messages,
                temperature=0, # Deterministic output
                max_tokens=100
            )
            optimized_query = response.choices[0].message.content.strip()
            print(f"REWRITER: '{user_query}' -> '{optimized_query}'")
            return optimized_query
        except Exception as e:
            print(f"Rewriter failed: {e}")
            return user_query


    def retrieve(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for ANY query.
        1. Rewrite query using LLM (if available)
        2. Semantic search with optimized query
        """
        if not self.collection:
            raise RuntimeError("Vector store not initialized. Run index_data() first.")

        # 1. INTELLIGENT REWRITE STEP
        search_query = self.rewrite_query(query)

        # Embed query
        query_embedding = self.embedder.encode([search_query])[0].tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved

    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM with retrieved context.
        This is the G in RAG - Generation.
        """
        if not self.groq_client:
            # Fallback: just return context
            return "LLM not available. Here's the relevant data:\n\n" + "\n\n---\n\n".join(
                [c["text"] for c in context]
            )

        # Build context string
        context_str = "\n\n---\n\n".join([c["text"] for c in context])

        prompt = f"""You are a helpful assistant answering questions about pancreatic cancer cell analysis data.
You have been given relevant data retrieved from a database. Answer the user's question based ONLY on this data.

RETRIEVED DATA:
{context_str}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the retrieved data above
- If the data doesn't contain the answer, say so
- Be specific and cite the relevant findings
- Include effect sizes, probabilities, and cell types when available
- Keep your answer focused and informative

ANSWER:"""

        response = self.groq_client.chat.completions.create(
            model="compound-beta-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )

        return response.choices[0].message.content

    def query(self, question: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Main RAG query function.

        This is TRUE RAG:
        1. Retrieve relevant documents via semantic search
        2. Generate answer using LLM with context

        Can answer ANY question about the data.
        """
        # Retrieve
        retrieved = self.retrieve(question, n_results=n_results)

        # Generate
        answer = self.generate_answer(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "num_sources": len(retrieved)
        }


def initialize_rag(force_reindex: bool = False) -> TrueRAG:
    """Initialize and index the RAG system."""
    rag = TrueRAG()
    rag.index_data(force=force_reindex)
    return rag


# CLI interface
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TRUE RAG SYSTEM - Ask ANY question about your data")
    print("=" * 60)

    # Initialize
    rag = TrueRAG()

    # Check if we need to index
    if "--reindex" in sys.argv:
        rag.index_data(force=True)
    else:
        rag.index_data(force=False)

    # Interactive mode
    print("\nReady! Ask any question about your cell analysis data.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            result = rag.query(question)
            print("\n" + "=" * 60)
            print("ANSWER:")
            print(result["answer"])
            print("\n" + "-" * 60)
            print(f"(Based on {result['num_sources']} retrieved sources)")
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")
