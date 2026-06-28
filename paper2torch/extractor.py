import re

def extract_sections(full_text: str) -> dict:
    sections = {
        "abstract": "",
        "introduction": "",
        "methodology": "",
        "architecture": "",
        "experiments": "",
        "conclusion": ""
    }
    
    section_patterns = {
        "abstract": r"abstract",
        "introduction": r"introduction",
        "methodology": r"(method|methodology|approach|proposed method)",
        "architecture": r"(architecture|model architecture|network architecture)",
        "experiments": r"(experiment|evaluation|results)",
        "conclusion": r"(conclusion|conclusions)"
    }
    
    lines = full_text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line is a section header
        if len(line.strip()) < 60:
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    current_section = section
                    break
        
        if current_section:
            sections[current_section] += line + "\n"
    
    return sections


def get_core_content(sections: dict) -> str:
    """
    LLM ko feed karne ke liye core content nikalta hai
    """
    priority = ["abstract", "methodology", "architecture", "introduction"]
    
    core = ""
    for section in priority:
        if sections.get(section):
            core += f"\n=== {section.upper()} ===\n"
            core += sections[section]
    
    # Limit to 8000 chars — Groq/Gemini context ke liye
    return core[:8000]