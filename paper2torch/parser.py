import fitz
import os

def extract_text_from_pdf(pdf_path: str) -> dict:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    full_text = ""
    pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page": page_num + 1,
            "text": text
        })
        full_text += text + "\n"
    
    doc.close()
    
    return {
        "full_text": full_text,
        "pages": pages,
        "total_pages": len(pages)
    }


def extract_relevant_sections(full_text: str) -> str:
    keywords = [
        "abstract", "introduction", "method", "methodology",
        "architecture", "model", "approach", "proposed",
        "network", "layer", "attention", "encoder", "decoder"
    ]
    
    lines = full_text.split('\n')
    relevant_lines = []
    capture = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(kw in line_lower for kw in keywords) and len(line.strip()) < 60:
            capture = True
        
        if capture:
            relevant_lines.append(line)
    
    return '\n'.join(relevant_lines)