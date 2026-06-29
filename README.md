# paper2torch

Convert any research paper (PDF) into runnable PyTorch code using LLMs.

## Demo

```bash
pip install paper2torch
paper2torch paper.pdf --title "Your Paper Title" --output ./generated
```

## What it does

1. Parses the PDF and extracts relevant sections
2. Sends content to LLM (Groq — free & fast)
3. Generates a complete PyTorch `nn.Module` implementation
4. Validates the generated code
5. Saves `model.py`, `config.py`, and `README_generated.md`

## Installation

```bash
pip install paper2torch
```

Set your Groq API key (free at [console.groq.com](https://console.groq.com)):

```bash
export GROQ_API_KEY="your_key_here"  # Linux/Mac
$env:GROQ_API_KEY="your_key_here"    # Windows PowerShell
```

## Usage

```bash
paper2torch paper.pdf --title "Paper Title" --output ./generated
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--title` | Paper title | Unknown Paper |
| `--output` | Output folder | ./output |

## Output

output/

├── model.py        # PyTorch nn.Module implementation

├── config.py       # Hyperparameters as dataclass

└── README_generated.md  # What was implemented

## Example

Input: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

Output: Complete Transformer implementation in PyTorch.

## Tested Papers

| Paper | Status |
|-------|--------|
| Attention is All You Need | ✅ |
| ResNet | 🔜 |
| BERT | 🔜 |

## Stack

- **LLM**: Groq (Llama 3.3 70B)
- **PDF Parsing**: PyMuPDF
- **CLI**: Click + Rich
- **Validation**: AST + PyTorch checks

## Roadmap

- [ ] Support for more architectures
- [ ] Training loop generation
- [ ] HuggingFace Spaces demo
- [ ] Multiple LLM provider support

## Contributing

PRs welcome. Open an issue first for major changes.

## License

MIT

---

Built by [Dhruv Kumar](https://linkedin.com/in/dhruv-kumar-67b26334b)
