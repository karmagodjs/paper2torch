import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_pytorch_code(core_content: str, paper_title: str = "Unknown") -> dict:

    # Step 1: Architecture extract karo
    arch_prompt = f"""
You are an expert ML engineer. Read this research paper content and extract the model architecture.

Paper content:
{core_content}

Return ONLY a JSON with these fields:
{{
    "model_name": "name of the model",
    "architecture_type": "transformer/cnn/rnn/mlp/other",
    "key_components": ["list", "of", "components"],
    "hyperparameters": {{"param_name": "value"}},
    "input_format": "description of input",
    "output_format": "description of output"
}}
Return only JSON, no explanation.
"""

    arch_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": arch_prompt}],
        max_tokens=1000
    )
    arch_text = arch_response.choices[0].message.content.strip()

    # Step 2: PyTorch code generate karo
    code_prompt = f"""
You are an expert PyTorch developer. Based on this research paper content, write a complete PyTorch implementation.

Paper content:
{core_content}

Rules:
1. Write a complete nn.Module class
2. Include __init__ and forward methods
3. Add clear comments explaining each component
4. Include a config dataclass at the top
5. Add a simple test at the bottom inside if __name__ == "__main__"
6. Use only standard PyTorch — no external libraries
7. Make it runnable as-is

Return ONLY Python code, no explanation, no markdown backticks.
"""

    code_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": code_prompt}],
        max_tokens=4000
    )
    code_text = code_response.choices[0].message.content.strip()

    if code_text.startswith("```"):
        lines = code_text.split('\n')
        code_text = '\n'.join(lines[1:-1])

    # Step 3: Config generate karo
    config_prompt = f"""
Based on this research paper, extract all hyperparameters and write a Python config dataclass.

Paper content:
{core_content[:2000]}

Return ONLY a Python dataclass with all hyperparameters found. Use @dataclass decorator.
No explanation, no markdown backticks.
"""

    config_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": config_prompt}],
        max_tokens=1000
    )
    config_text = config_response.choices[0].message.content.strip()

    if config_text.startswith("```"):
        lines = config_text.split('\n')
        config_text = '\n'.join(lines[1:-1])

    return {
        "architecture_info": arch_text,
        "model_code": code_text,
        "config_code": config_text
    }