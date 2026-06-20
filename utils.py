import re

def clean_markdown(text: str) -> str:
    text = re.sub(r'(\*{1,3})(.*?)\1', r'\2', text)
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('`', '')
    text = re.sub(r'[-*_]{3,}', '', text)
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    return text.strip()

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE
    )

    return emoji_pattern.sub("", text)
    
def truncate(text: str, limit: int = 500) -> str:
    """Tronca il testo se è troppo lungo, aggiungendo '...' alla fine."""
    return text[:limit] + "..." if len(text) > limit else text
    
# Console colors
class Colors:
    USER = '\033[94m'
    PARTIAL = '\033[93m'
    ASSISTANT = '\033[92m'
    ERROR = '\033[91m'
    RESET = '\033[0m'