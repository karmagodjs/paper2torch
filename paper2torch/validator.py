import ast
import torch
import sys


def validate_syntax(code: str) -> dict:
    """
    Python syntax check karta hai
    """
    try:
        ast.parse(code)
        return {
            "valid": True,
            "error": None
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}"
        }


def validate_pytorch(code: str) -> dict:
    """
    PyTorch import aur basic structure check karta hai
    """
    issues = []
    
    # torch import check
    if "import torch" not in code:
        issues.append("torch not imported")
    
    # nn.Module check
    if "nn.Module" not in code:
        issues.append("No nn.Module class found")
    
    # forward method check
    if "def forward" not in code:
        issues.append("No forward method found")
    
    # __init__ check
    if "def __init__" not in code:
        issues.append("No __init__ method found")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def validate_code(code: str) -> dict:
    """
    Full validation — syntax + pytorch structure
    """
    syntax_result = validate_syntax(code)
    
    if not syntax_result["valid"]:
        return {
            "passed": False,
            "syntax": syntax_result,
            "pytorch": None,
            "summary": f"Syntax error: {syntax_result['error']}"
        }
    
    pytorch_result = validate_pytorch(code)
    
    passed = pytorch_result["valid"]
    
    summary = "All checks passed!" if passed else f"Issues: {', '.join(pytorch_result['issues'])}"
    
    return {
        "passed": passed,
        "syntax": syntax_result,
        "pytorch": pytorch_result,
        "summary": summary
    }