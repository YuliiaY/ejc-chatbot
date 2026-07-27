import asyncio
import io

import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SEED_URLS = [
    "https://eja.net/",
    "https://ejc2026.org/"
]

DB_PATH = "./chroma_db"

def extract_pdf_text_from_url(pdf_url: str) -> str:
    """Download a PDF in memory and extract its text content."""
    try:
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        extracted_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(f"--- Page {i+1} ---\n{page_text}")
                
        return "\n\n".join(extracted_text)
    except Exception as e:
        print(f"Failed to process PDF at {pdf_url}: {e}")
        return ""

async def crawl_all_and_index():
    print("Starting web and PDF crawl process...")
    
    # max_depth=5 & max_pages=300 covers all pages without crashing Playwright
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=10,            
        max_pages=300,          
        include_external=False  
    )

    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        deep_crawl_strategy=deep_crawl_strategy,
        word_count_threshold=0,  
        stream=True             # Stream results as they complete to prevent silent hanging
    )

    all_crawled_pages = []
    discovered_pdf_urls = set()

    async with AsyncWebCrawler() as crawler:
        for seed_url in SEED_URLS:
            print(f"\nInitiating crawl for: {seed_url}")
            
            results_count = 0
            async for result in await crawler.arun(url=seed_url, config=crawl_config):
                results_count += 1
                if result and result.success:
                    print(f"  └─ Captured [{results_count}]: {result.url}")
                    content = result.markdown if result.markdown else f"[Page without text: {result.url}]"
                    all_crawled_pages.append({
                        "content": content,
                        "url": result.url,
                        "title": result.metadata.get("title", result.url) if result.metadata else result.url
                    })

                    # 1. Look for PDF links via metadata
                    if hasattr(result, 'links') and result.links:
                        for link in result.links.get("internal", []) + result.links.get("external", []):
                            href = link.get("href", "")
                            if href.lower().endswith(".pdf"):
                                discovered_pdf_urls.add(href)

                    # 2. Fallback PDF link discovery from raw HTML
                    if result.html:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag['href']
                            if href.lower().endswith('.pdf'):
                                if href.startswith('http'):
                                    discovered_pdf_urls.add(href)
                                elif href.startswith('/'):
                                    base_domain = seed_url.rstrip('/')
                                    discovered_pdf_urls.add(f"{base_domain}{href}")

    print(f"\nCrawl Phase Complete! Total Web Pages collected: {len(all_crawled_pages)}")
    print(f"Found {len(discovered_pdf_urls)} unique PDF files to process.")

    # Process downloaded PDFs
    for pdf_url in discovered_pdf_urls:
        print(f"Extracting text from PDF: {pdf_url}")
        pdf_text = extract_pdf_text_from_url(pdf_url)
        if pdf_text:
            all_crawled_pages.append({
                "content": pdf_text,
                "url": pdf_url,
                "title": f"PDF Document ({pdf_url.split('/')[-1]})"
            })

    if not all_crawled_pages:
        print("Error: No pages were collected. Check your internet connection or target URLs.")
        return

    # Chunking pipeline
    print("\nSplitting content into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    metadatas = []

    for page in all_crawled_pages:
        chunks = text_splitter.split_text(page["content"])
        if not chunks:
            chunks = [f"[Captured page with no text: {page['url']}]"]

        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({
                "source": page["url"],
                "title": page["title"]
            })

    print(f"Total text chunks created across Web + PDF: {len(documents)}")

    # Embedding and batch storage in Chroma DB
    print("\nGenerating embeddings ...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    BATCH_SIZE = 50 
    vectorstore = None

    total_batches = (len(documents) - 1) // BATCH_SIZE + 1

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]
        
        current_batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {current_batch_num} / {total_batches} ({len(batch_docs)} chunks)...")

        try:
            if vectorstore is None:
                vectorstore = Chroma.from_texts(
                    texts=batch_docs,
                    embedding=embeddings,
                    metadatas=batch_meta,
                    persist_directory=DB_PATH
                )
            else:
                vectorstore.add_texts(
                    texts=batch_docs,
                    metadatas=batch_meta
                )
        except Exception as e:
            print(f"Batch {current_batch_num} hit an error: {e}. Retrying item-by-item...")
            for single_doc, single_meta in zip(batch_docs, batch_meta):
                if vectorstore is None:
                    vectorstore = Chroma.from_texts(
                        texts=[single_doc],
                        embedding=embeddings,
                        metadatas=[single_meta],
                        persist_directory=DB_PATH
                    )
                else:
                    vectorstore.add_texts(
                        texts=[single_doc],
                        metadatas=[single_meta]
                    )

    print(f"\nSuccess! Database built and saved to '{DB_PATH}'.")

if __name__ == "__main__":
    asyncio.run(crawl_all_and_index())