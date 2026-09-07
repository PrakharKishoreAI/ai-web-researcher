import os

import streamlit as st
from dotenv import load_dotenv
from firecrawl import Firecrawl
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Web Researcher",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# Get Groq API key
# --------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY was not found in .env")
    st.stop()


# --------------------------------------------------
# Initialize LLM through LangChain
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=groq_api_key
)


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def chunk_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🔎 AI Web Researcher")

st.write(
    "Research any topic using live web information and AI."
)


query = st.text_input(
    "What do you want to research?",
    placeholder="e.g. Latest developments in Generative AI"
)


# --------------------------------------------------
# Research
# --------------------------------------------------

if st.button("🚀 Research"):

    if not query:

        st.warning("Please enter a research topic.")

    else:

        firecrawl_api_key = os.getenv(
            "FIRECRAWL_API_KEY"
        )

        if not firecrawl_api_key:

            st.error(
                "FIRECRAWL_API_KEY was not found in .env"
            )

            st.stop()


        # --------------------------------------------------
        # Firecrawl
        # --------------------------------------------------

        firecrawl = Firecrawl(
            api_key=firecrawl_api_key
        )


        with st.spinner("Searching the web..."):

            results = firecrawl.search(
                query,
                limit=5
            )

        st.success("Web search completed!")


        # --------------------------------------------------
        # ChromaDB
        # --------------------------------------------------

        chroma_client = chromadb.Client()

        collection = chroma_client.get_or_create_collection(
            name="web_research"
        )


        all_chunks = []
        all_embeddings = []
        all_metadata = []


        # --------------------------------------------------
        # Search + scrape
        # --------------------------------------------------

        st.subheader("📄 Research Sources")


        for i, result in enumerate(
            results.web,
            start=1
        ):

            url = result.url

            with st.spinner(
                f"Reading source {i}..."
            ):

                try:

                    page = firecrawl.scrape(url)

                    content = page.markdown or ""


                    if content:

                        chunks = chunk_text(
                            content
                        )


                        embeddings = (
                            embedding_model
                            .encode(chunks)
                            .tolist()
                        )


                        for chunk, embedding in zip(
                            chunks,
                            embeddings
                        ):

                            all_chunks.append(chunk)

                            all_embeddings.append(
                                embedding
                            )

                            all_metadata.append(
                                {
                                    "source": url,
                                    "title": result.title
                                }
                            )


                        with st.expander(
                            f"{i}. {result.title}"
                        ):

                            st.write(url)

                            st.write(
                                f"Created {len(chunks)} chunks"
                            )


                    else:

                        st.warning(
                            f"No readable content found for source {i}."
                        )


                except Exception as e:

                    st.warning(
                        f"Could not read source {i}: {e}"
                    )


        # --------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------

        if all_chunks:

            st.subheader(
                "🗄️ Semantic Retrieval"
            )


            ids = [
                f"chunk_{i}"
                for i in range(
                    len(all_chunks)
                )
            ]


            collection.add(
                ids=ids,
                documents=all_chunks,
                embeddings=all_embeddings,
                metadatas=all_metadata
            )


            st.success(
                f"Stored {len(all_chunks)} chunks in ChromaDB."
            )


            # Query embedding
            query_embedding = (
                embedding_model
                .encode([query])
                .tolist()
            )


            # Retrieve relevant chunks
            retrieved = collection.query(
                query_embeddings=query_embedding,
                n_results=min(
                    5,
                    len(all_chunks)
                )
            )


            # --------------------------------------------------
            # Prepare context
            # --------------------------------------------------

            documents = retrieved["documents"][0]

            metadata = retrieved["metadatas"][0]


            context_parts = []

            for i, document in enumerate(
                documents
            ):

                source_title = metadata[i]["title"]

                source_url = metadata[i]["source"]

                context_parts.append(
                    f"""
SOURCE:
{source_title}

URL:
{source_url}

CONTENT:
{document}
"""
                )


            context = "\n\n".join(
                context_parts
            )


            # --------------------------------------------------
            # Generate research report
            # --------------------------------------------------

            prompt = f"""
You are an AI web research assistant.

Research question:
{query}

Below is information retrieved from live web sources:

{context}

Create a clear research report answering the user's question.

Requirements:
- Directly answer the research question.
- Use headings and bullet points where useful.
- Summarize the important findings.
- Use only information supported by the retrieved sources.
- Do not invent facts.
- Keep the language clear and professional.
"""


            with st.spinner(
                "Generating research report..."
            ):

                try:

                    response = llm.invoke(
                        prompt
                    )

                    report = response.content

                except Exception as e:

                    st.error(
                        f"Could not generate the report: {e}"
                    )

                    st.stop()


            # --------------------------------------------------
            # Final report
            # --------------------------------------------------

            st.subheader(
                "🤖 Research Report"
            )

            st.markdown(
                report
            )


            # --------------------------------------------------
            # Sources
            # --------------------------------------------------

            st.subheader(
                "🌐 Sources"
            )


            shown_sources = set()


            for item in metadata:

                title = item["title"]

                url = item["source"]

                if url not in shown_sources:

                    st.markdown(
                        f"- [{title}]({url})"
                    )

                    shown_sources.add(url)


        else:

            st.error(
                "No usable web content was found."
            )