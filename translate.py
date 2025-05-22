import os
import sys
import requests
import unicodedata
import time
import json
import shutil
import threading
import subprocess
import re
from datetime import timedelta, datetime
from pathlib import Path

# Check and install required dependencies
def check_dependencies():
    """Verify and install required dependencies"""
    try:
        # First check if the packages are installed
        import importlib.util
        nltk_spec = importlib.util.find_spec("nltk")
        docx_spec = importlib.util.find_spec("docx")  # python-docx imports as docx
        lxml_spec = importlib.util.find_spec("lxml")
        
        if all([nltk_spec, docx_spec, lxml_spec]):
            # Then try to import them to ensure they work
            import nltk
            import docx
            import lxml.etree
            print("Required dependencies are already installed.")
            return True
        else:
            raise ImportError("One or more required packages not found")
    except ImportError:
        print("Installing required dependencies...")
        try:
            import pip
            pip.main(['install', 'nltk', 'python-docx', 'lxml'])
            # After installing, try importing them again
            import nltk
            import docx
            import lxml.etree
            print("Successfully installed dependencies.")
            return True
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            print("Please run: pip install nltk python-docx lxml")
            return False

# Check dependencies first
if not check_dependencies():
    sys.exit(1)

# Now we can safely import these
import nltk
import docx
from lxml import etree

# Download NLTK data
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")
    print("Some sentence splitting may not work correctly.")

# Default HuggingFace API endpoints
HF_API_URL = os.getenv("HF_API_URL", "https://bz2eki98bvwdoh9l.us-east4.gcp.endpoints.huggingface.cloud")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2") # Default to Mistral model if not specified
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Please set the HF_TOKEN environment variable")
    sys.exit(1)

# Translation template environment variables
TRANSLATE_TEMPLATE_En = os.getenv("TRANSLATE_TEMPLATE_En", "")
TRANSLATE_TEMPLATE_Ja = os.getenv("TRANSLATE_TEMPLATE_Ja", "")

def check_translation_templates():
    """Check for translation templates in environment variables and set up defaults if needed"""
    global TRANSLATE_TEMPLATE_En, TRANSLATE_TEMPLATE_Ja
    
    # Check if environment variable is set
    if not TRANSLATE_TEMPLATE_En:
        print("\nNOTE: TRANSLATE_TEMPLATE_En environment variable not set.")
        print("For best results, set this environment variable with a template like:")
        print("-------------------------------------------------------------------")
        print('TRANSLATE_TEMPLATE_En="[INST] You are a professional Japanese translator. Your task is to translate the following English text into natural, fluent Japanese.\n\nTranslation style: Young adult fantasy novel for vertical writing (tategaki).\nExpress subtle wonder, emotional depth, and a slightly melancholic tone.\nUse appropriate Japanese punctuation (「」for quotes, 。for periods, etc).\nUse natural Japanese expressions rather than literal translations.\n\nIMPORTANT: Return ONLY the Japanese translation. No explanations or English text.\n\nText to translate: {text} [/INST]"')
        print("-------------------------------------------------------------------")
        print("Using a default template for now. Set the environment variable for best results.\n")
    else:
        print("Found TRANSLATE_TEMPLATE_En in environment variables.")
        
    # Check template format
    if TRANSLATE_TEMPLATE_En and "{text}" not in TRANSLATE_TEMPLATE_En:
        print("WARNING: TRANSLATE_TEMPLATE_En must contain {text} placeholder. Using default template.")
        TRANSLATE_TEMPLATE_En = ""
        
    return True

# Check translation templates
check_translation_templates()

# Allow fallback to other models if the first one fails
BACKUP_MODELS = [
    "meta-llama/Llama-2-70b-chat-hf",
    "gemini/gemini-pro",
    "google/gemma-7b-it",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to test the API connection
def test_api_connection():
    """Test the API connection and return diagnostic information"""
    print(f"Testing API connection to {HF_API_URL}...")
    short_prompt = "Translate this to Japanese: Hello world."
    payload = {
        "inputs": short_prompt,
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False,
        },
    }
    
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=10)
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"API Response structure: {json.dumps(data, indent=2)[:500]}...")
                return True
            except json.JSONDecodeError:
                print(f"Could not parse API response as JSON. Response text: {response.text[:500]}...")
                return False
        elif response.status_code == 401 or response.status_code == 403:
            print("Authentication error. Please check your HF_TOKEN.")
            return False
        elif response.status_code == 404:
            print("API endpoint not found. Please verify the API URL.")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}, Response: {response.text[:500]}...")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

def test_translation_template():
    """Test the translation with the current template"""
    global TRANSLATE_TEMPLATE_En
    
    print("\nTesting translation with current template...")
    print(f"Using API endpoint: {HF_API_URL}")
    print(f"Using model: {HF_MODEL}")
    
    # Simple test phrase
    test_text = "The old maple tree whispered secrets as the autumn breeze passed through its branches."
    
    try:
        # Translate the test phrase
        translated = translate_text(test_text)
        
        # Check if translation succeeded
        has_jp_chars = any(0x3000 <= ord(char) <= 0x9FFF for char in translated)
        
        if has_jp_chars:
            print("\n✓ Translation test successful!")
            print(f"Input:  {test_text}")
            print(f"Output: {translated}")
            return True
        else:
            print("\n✗ Translation test failed. No Japanese characters in output.")
            print(f"Output received: {translated}")
            return False
    except Exception as e:
        print(f"\n✗ Translation test error: {e}")
        return False

OUTPUT_FILE = "JP.txt"
CHECKPOINT_FILE = "translation_checkpoint.json"
BACKUP_DIR = "translate"
GCS_BUCKET = "gs://cdn-othertales-co/"
BACKUP_INTERVAL = 120  # Backup every 2 minutes

# Create backup directory if it doesn't exist
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR, exist_ok=True)

# Flag to control the backup thread
backup_thread_active = False

def cloud_backup_thread():
    """Background thread that backs up the translation file to GCS every 2 minutes"""
    global backup_thread_active
    
    while backup_thread_active:
        try:
            if os.path.exists(OUTPUT_FILE):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create backup filename with timestamp and original filename
                filename = Path(OUTPUT_FILE).name
                base_name = Path(filename).stem
                extension = Path(filename).suffix
                timestamped_filename = f"{base_name}_{timestamp}{extension}"
                
                # Backup to the main file (overwrite) and also to a timestamped version
                print(f"[{timestamp}] Backing up {OUTPUT_FILE} to {GCS_BUCKET}")
                
                # Execute gsutil command to copy the file to Google Cloud Storage
                # First, copy to the main filename (overwriting previous version)
                result = subprocess.run(
                    ["gsutil", "cp", OUTPUT_FILE, GCS_BUCKET],
                    capture_output=True, 
                    text=True
                )
                
                # Then, copy to a timestamped filename (for versioning)
                timestamped_result = subprocess.run(
                    ["gsutil", "cp", OUTPUT_FILE, f"{GCS_BUCKET}{timestamped_filename}"],
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0 and timestamped_result.returncode == 0:
                    print(f"[{timestamp}] Backup successful")
                else:
                    if result.returncode != 0:
                        print(f"[{timestamp}] Main backup failed: {result.stderr}")
                    if timestamped_result.returncode != 0:
                        print(f"[{timestamp}] Timestamped backup failed: {timestamped_result.stderr}")
        except Exception as e:
            print(f"Error during cloud backup: {e}")
        
        # Sleep for the specified interval
        time.sleep(BACKUP_INTERVAL)

def start_backup_thread():
    """Start the background backup thread"""
    global backup_thread_active
    
    # Check if gsutil is available
    try:
        result = subprocess.run(
            ["gsutil", "version"], 
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Warning: gsutil not found or not configured. Cloud backups will be disabled.")
            print("Please install and configure Google Cloud SDK if you want cloud backups.")
            return False
    except FileNotFoundError:
        print("Warning: gsutil not found. Cloud backups will be disabled.")
        print("Please install Google Cloud SDK if you want cloud backups.")
        return False
    
    backup_thread_active = True
    backup_thread = threading.Thread(target=cloud_backup_thread, daemon=True)
    backup_thread.start()
    print("Cloud backup thread started. Will backup every 2 minutes.")
    return True

def stop_backup_thread():
    """Stop the background backup thread"""
    global backup_thread_active
    backup_thread_active = False
    print("Cloud backup thread stopped.")

def normalize_punctuation(text):
    # First apply NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Replace Western punctuation with Japanese equivalents suitable for Tategaki
    replacements = {
        '.': '。',    # Western period to Japanese full stop
        ',': '、',    # Western comma to Japanese comma
        '!': '！',    # Exclamation mark to full-width
        '?': '？',    # Question mark to full-width
        '(': '（',    # Opening parenthesis to full-width
        ')': '）',    # Closing parenthesis to full-width
        '"': '「',    # Opening quote to Japanese opening quote
        '"': '」',    # Closing quote to Japanese closing quote
        "'": '『',    # Opening single quote to Japanese nested opening quote
        "'": '』',    # Closing single quote to Japanese nested closing quote
        ':': '：',    # Colon to full-width
        ';': '；',    # Semicolon to full-width
        '-': 'ー',    # Hyphen to Japanese long vowel mark
        '...': '…',   # Ellipsis to Japanese ellipsis
        '–': '—',     # En dash to Em dash
    }
    
    # Apply replacements
    for western, japanese in replacements.items():
        text = text.replace(western, japanese)
    
    # Handle special cases for quotes that may be ASCII quotes
    text = text.replace('"', '「').replace('"', '」')
    text = text.replace("'", '『').replace("'", '』')
    
    # Additional specific replacements for common patterns
    text = text.replace('--', '—')
    
    return text

def clean_translation_output(text):
    """
    Clean up the translation output to extract only the Japanese text.
    Removes formatting markers, prefixes, and other non-translation content.
    
    More robust handling of different response types:
    1. Structured responses with markers (e.g. "**Japanese Translation:**")
    2. Mixed language responses (extract only Japanese parts)
    3. Pure English responses (return error message)
    """
    # Helper function to check if a character is Japanese
    def is_japanese_char(char):
        code = ord(char)
        return (0x3000 <= code <= 0x9FFF or   # CJK unified ideographs, Hiragana, Katakana
                0xF900 <= code <= 0xFAFF or   # CJK compatibility ideographs
                0xFF00 <= code <= 0xFFEF)      # Half-width and full-width forms
    
    # Helper function to check if a character is Japanese punctuation
    def is_japanese_punct(char):
        return char in "「」『』（）！？。、：；"
    
    # First, check if there are any Japanese characters in the response
    has_japanese = any(is_japanese_char(char) for char in text)
    
    if not has_japanese:
        print(f"WARNING: No Japanese characters found in the response: {text[:100]}...")
        return "［翻訳エラー］"  # Japanese for [translation error]
    
    # Calculate the Japanese character ratio
    jp_char_count = sum(1 for char in text if is_japanese_char(char) or is_japanese_punct(char))
    total_chars = len(text.strip())
    jp_ratio = jp_char_count / total_chars if total_chars > 0 else 0
    
    # Very clean response (mostly Japanese) - just do basic cleanup
    if jp_ratio > 0.7:  # If more than 70% is Japanese, it's likely a clean response
        # Just strip and remove unnecessary quotes
        cleaned_text = text.strip()
        if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or \
           (cleaned_text.startswith("'") and cleaned_text.endswith("'")):
            cleaned_text = cleaned_text[1:-1].strip()
        return cleaned_text
    
    # Common patterns to remove from model output
    patterns = [
        # Match anything with "Original Sentence:" or "Japanese Translation:" markers
        r'\*\*Original Sentence[：:]\*\*.*?(?=\*\*Japanese Translation[：:]\*\*|\Z)',
        r'\*\*Japanese Translation[：:]\*\*\s*',
        r'\*\*[^*]+?\*\*',  # Any **text** format markers
        r'##\s*Japanese\s*Translation[：:]\s*',  # Section headers
        r'##\s*[^#]+',  # Any ## header format
        r'Japanese Translation[：:]',  # Plain text markers
        r'Translation[：:]',  # More general translation markers
        r'.*?(?=日本語|[ぁ-んァ-ン])',  # Match everything up to first Japanese character
    ]
    
    # Check if the text contains markers that indicate a structured response
    has_markers = any(re.search(pattern, text) for pattern in [
        r'\*\*Japanese Translation', 
        r'##\s*Japanese\s*Translation',
        r'Japanese Translation[：:]',
        r'Translation[：:]'
    ])
    
    # Try to extract Japanese text using different methods
    extracted_text = None
    
    # 1. MARKER-BASED EXTRACTION
    if has_markers:
        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
        
        # If we got good Japanese content, use it
        jp_char_count = sum(1 for char in cleaned_text if is_japanese_char(char))
        if jp_char_count > 10:  # Arbitrary threshold for "enough" Japanese text
            extracted_text = cleaned_text
    
    # 2. SENTENCE-BASED EXTRACTION
    if not extracted_text:
        # Find continuous Japanese segments with Japanese punctuation
        japanese_segments = re.findall(r'[「」『』（）！？。、]?[^\n\r]+?[「」『』（）！？。、]', text)
        
        if japanese_segments:
            # Filter segments to only those with Japanese characters
            japanese_segments = [
                seg for seg in japanese_segments 
                if any(is_japanese_char(char) for char in seg)
            ]
            
            if japanese_segments:
                extracted_text = "".join(japanese_segments)
    
    # 3. LINE-BASED EXTRACTION
    if not extracted_text:
        lines = text.split('\n')
        japanese_lines = []
        
        for line in lines:
            # Count Japanese characters in this line
            jp_chars = sum(1 for c in line if is_japanese_char(c))
            
            # If there are Japanese characters and the line is mostly Japanese or contains sentences
            if jp_chars > 0 and ('。' in line or '！' in line or '？' in line or jp_chars / len(line) > 0.3):
                # Extract just Japanese segments within this line
                jp_segments = re.findall(r'[「」『』（）！？。、]?[^a-zA-Z\(\)]+[「」『』（）！？。、]?', line)
                if jp_segments:
                    japanese_lines.append("".join(jp_segments))
                else:
                    japanese_lines.append(line)
        
        if japanese_lines:
            extracted_text = '\n'.join(japanese_lines)
    
    # 4. CHARACTER-BY-CHARACTER EXTRACTION (last resort)
    if not extracted_text:
        # Extract continuous Japanese characters sequences
        japanese_parts = []
        current_part = []
        
        for char in text:
            if is_japanese_char(char) or is_japanese_punct(char) or char.isspace():
                current_part.append(char)
            elif current_part:  # End of a Japanese segment
                segment = ''.join(current_part).strip()
                if any(is_japanese_char(c) for c in segment):
                    japanese_parts.append(segment)
                current_part = []
        
        # Don't forget the last segment
        if current_part:
            segment = ''.join(current_part).strip()
            if any(is_japanese_char(c) for c in segment):
                japanese_parts.append(segment)
        
        if japanese_parts:
            extracted_text = ' '.join(japanese_parts)
    
    # Use the extracted text if we found any, otherwise use the original
    final_text = extracted_text if extracted_text else text
    
    # Final cleanup
    final_text = final_text.strip()
    
    # Remove any remaining English phrases in parentheses - common in translations
    final_text = re.sub(r'\([^)]*[a-zA-Z][^)]*\)', '', final_text)
    
    # If final text is still wrapped in quotes, remove them
    if (final_text.startswith('"') and final_text.endswith('"')) or \
       (final_text.startswith("'") and final_text.endswith("'")):
        final_text = final_text[1:-1].strip()
    
    # If we don't have enough Japanese characters in the final result, return error
    jp_char_count = sum(1 for char in final_text if is_japanese_char(char))
    if jp_char_count < 5:  # Arbitrary threshold for "enough" Japanese text
        print(f"WARNING: Too few Japanese characters in cleaned output: {final_text}")
        return "［翻訳エラー］"  # Japanese for [translation error]
    
    return final_text

def build_translation_prompt(text_to_translate, model_name=None):
    """Build an appropriate translation prompt based on model type"""
    global TRANSLATE_TEMPLATE_En, TRANSLATE_TEMPLATE_Ja
    
    # If environment variable template is available, use it
    if TRANSLATE_TEMPLATE_En:
        # Replace {text} with the actual text to translate
        return TRANSLATE_TEMPLATE_En.replace("{text}", text_to_translate)
    # If Japanese template is available and we detect Japanese input
    elif TRANSLATE_TEMPLATE_Ja and any(0x3000 <= ord(char) <= 0x9FFF for char in text_to_translate):
        return TRANSLATE_TEMPLATE_Ja.replace("{text}", text_to_translate)
    
    # Otherwise, use model-specific templates
    # For Mistral and similar instruction models
    elif model_name and ("mistral" in model_name.lower() or "llama" in model_name.lower() or 
                       "gemma" in model_name.lower() or "mixtral" in model_name.lower()):
        return (
            "[INST] You are a professional Japanese translator. Translate this English text to Japanese:\n\n"
            "Translation style: Young adult fantasy novel for vertical writing (tategaki). "
            "Express subtle wonder, emotional depth, and a slightly melancholic tone.\n"
            "IMPORTANT: Output Japanese text ONLY. No explanations or English text.\n\n"
            f"Text to translate: {text_to_translate} [/INST]"
        )
    # For Claude-like models
    elif model_name and "claude" in model_name.lower():
        return (
            "Human: You are a professional Japanese translator. Translate this English text to Japanese:\n\n"
            "Translation style: Young adult fantasy novel for vertical writing (tategaki). "
            "Express subtle wonder, emotional depth, and a slightly melancholic tone.\n"
            "IMPORTANT: Output Japanese text ONLY. No explanations or English text.\n\n"
            f"Text to translate: {text_to_translate}\n\n"
            "Assistant: "
        )
    # For GPT-like models or default
    else:
        return (
            "あなたは優秀な日本語翻訳者です。次の英語の文章を日本語に翻訳してください。\n\n"
            "翻訳スタイルは日本の若者向けファンタジー小説（縦書き用）です。微妙な不思議さ、感情の深さ、"
            "
