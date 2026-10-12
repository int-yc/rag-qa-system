
import glob
import os
from typing import List

from config.settings import RETRIEVER_CONFIG, PATH_CONFIG
from utils.logger import logger

class DocumentLoader:

    def __init__(self, data_dir: str = None):

        self.data_dir = data_dir or PATH_CONFIG["data_dir"]

    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            texts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"读取 PDF 失败 {path}: {e}")
            return ""

    def load_all(self) -> List[str]:
        all_texts = []

        patterns = ["*.txt", "*.pdf", "*.md"]
        for pattern in patterns:
            for path in glob.glob(os.path.join(self.data_dir, pattern)):
                logger.info(f"读取文档：{os.path.basename(path)}")
                if path.endswith(".pdf"):
                    text = self._read_pdf(path)
                else:
                    text = self._read_txt(path)
                if text.strip():
                    all_texts.append(text.strip())
        logger.info(f"共读取 {len(all_texts)} 个文档")
        return all_texts

class TextSplitter:

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or RETRIEVER_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or RETRIEVER_CONFIG["chunk_overlap"]

    def split(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap
        return chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        all_chunks = []
        for doc in documents:
            chunks = self.split(doc)
            all_chunks.extend(chunks)
        logger.info(f"切分完成，共 {len(all_chunks)} 个文本块")
        return all_chunks
