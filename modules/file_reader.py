"""
File Reader Module
Supports reading text from .md, .txt, and .docx files
"""

import os
from pathlib import Path


def read_markdown(file_path: str) -> str:
    """Read markdown file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_text(file_path: str) -> str:
    """Read plain text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_docx(file_path: str) -> str:
    """Read DOCX file"""
    try:
        from docx import Document

        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        return "\n\n".join(text)
    except ImportError:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: pip install python-docx"
        )


def read_file(file_path: str) -> str:
    """
    Read file based on extension
    Supports: .md, .txt, .docx

    Args:
        file_path: Path to the file

    Returns:
        File content as string

    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".md":
        return read_markdown(file_path)
    elif ext == ".txt":
        return read_text(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}\n" "Supported formats: .md, .txt, .docx"
        )


def get_file_info(file_path: str) -> dict:
    """Get file information"""
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "path": file_path,
        "size": stat.st_size,
        "extension": Path(file_path).suffix.lower(),
    }


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            content = read_file(file_path)
            info = get_file_info(file_path)

            print(f"File: {info['name']}")
            print(f"Size: {info['size']} bytes")
            print(f"Extension: {info['extension']}")
            print(f"\nContent preview:")
            print(content[:200] + "..." if len(content) > 200 else content)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python file_reader.py <file_path>")
