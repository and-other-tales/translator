import os
import sys
import requests
import nltk
import docx
import unicodedata
import time
import re
from datetime import timedelta

nltk.download("punkt", quiet=True)

HF_API_URL = "https://bz2eki98bvwdoh9l.us-east4.gcp.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Please set the HF_TOKEN environment variable")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

OUTPUT_FILE = "JP.txt"

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

def clean_english_text(text):
    """Remove any English text or markdown that might be in the translation."""
    # Remove markdown symbols
    text = re.sub(r'[#*_`]', '', text)
    
    # Remove English notes, explanations, etc.
    text = re.sub(r'Notes:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Explanation:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Additional .*:', '', text, flags=re.IGNORECASE)
    
    # Remove any lines that are pure English
    lines = text.split('\n')
    japanese_lines = []
    for line in lines:
        # Keep lines that have at least some Japanese characters
        if re.search(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', line):
            japanese_lines.append(line)
    
    return '\n'.join(japanese_lines)

def translate_text(text_to_translate):
    prompt = (
        "Translate the following English text into casual Japanese suitable for a young adult magical realism novel. "
        "Return ONLY the translated Japanese text without any comments, notes, explanations, or formatting. "
        "Do not include any English text in your response. Provide only the bare Japanese translation:\n\n"
        + text_to_translate
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "return_full_text": False,
        },
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    data = response.json()
    raw_text = data[0]["generated_text"].strip()
    
    # Clean any English or markdown that might be included
    clean_text = clean_english_text(raw_text)
    
    # Normalize punctuation for Tategaki
    final_text = normalize_punctuation(clean_text)
    
    return final_text

def calculate_eta(start_time, current_progress, total_items):
    elapsed = time.time() - start_time
    if current_progress == 0:  # Avoid division by zero
        return "calculating..."
    
    progress_ratio = current_progress / total_items
    if progress_ratio == 0:  # Avoid division by zero
        return "calculating..."
    
    total_time_estimate = elapsed / progress_ratio
    remaining_time = total_time_estimate - elapsed
    
    # Format as HH:MM:SS
    return str(timedelta(seconds=int(remaining_time)))

def save_to_file(text, append=True):
    """Save text to file with proper flushing to ensure it's written immediately"""
    mode = "a" if append else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk
    print(f"Text written to file: {text}")

def main():
    docx_path = input("Enter the full path to your DOCX manuscript: ").strip()
    if not os.path.isfile(docx_path):
        print(f"File not found: {docx_path}")
        sys.exit(1)

    # Clear output file at start
    save_to_file("", append=False)

    doc = docx.Document(docx_path)

    # First pass: Count all meaningful content units (sentences and titles)
    total_content_units = 0
    total_paragraphs = 0
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # skip empty paragraphs
        
        if text.startswith("###") or text.startswith("##"):
            # Titles count as 1 content unit each
            total_content_units += 1
        else:
            # Regular paragraphs: count sentences
            sentences = nltk.sent_tokenize(text)
            total_content_units += len(sentences)
            total_paragraphs += 1

    print(f"Document contains {total_content_units} content units to translate ({total_paragraphs} paragraphs)")
    
    # Separate counters for different content types
    paragraph_number = 0
    manuscript_title_count = 0
    chapter_title_count = 0
    content_units_processed = 0
    start_time = time.time()
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # skip empty paragraphs

        if text.startswith("###"):  # Manuscript title line
            manuscript_title_count += 1
            to_translate = text[3:].strip()
            
            # Update progress
            content_units_processed += 1
            percent_done = (content_units_processed / total_content_units) * 100
            eta = calculate_eta(start_time, content_units_processed, total_content_units)
            
            print(f"Translating manuscript title {manuscript_title_count}: {to_translate}")
            
            try:
                jp_translation = translate_text(to_translate)
                print(f"Done: {content_units_processed}/{total_content_units} ({percent_done:.1f}%) ETA: {eta} [{jp_translation}]")
                
                # Write ONLY the clean Japanese translation - no markdown prefix
                save_to_file(f"{jp_translation}\n\n")
            except Exception as e:
                print(f"Error translating manuscript title {manuscript_title_count}: {e}")
                # Skip writing errors to output file for clean printing

        elif text.startswith("##"):  # Chapter title line
            chapter_title_count += 1
            to_translate = text[2:].strip()
            
            # Update progress
            content_units_processed += 1
            percent_done = (content_units_processed / total_content_units) * 100
            eta = calculate_eta(start_time, content_units_processed, total_content_units)
            
            print(f"Translating chapter title {chapter_title_count}: {to_translate}")
            
            try:
                jp_translation = translate_text(to_translate)
                print(f"Done: {content_units_processed}/{total_content_units} ({percent_done:.1f}%) ETA: {eta} [{jp_translation}]")
                
                # Write ONLY the clean Japanese translation - no markdown prefix
                save_to_file(f"{jp_translation}\n\n")
            except Exception as e:
                print(f"Error translating chapter title {chapter_title_count}: {e}")
                # Skip writing errors to output file for clean printing

        else:
            # Only increment paragraph count for regular paragraphs
            paragraph_number += 1
            print(f"Processing paragraph {paragraph_number}...")
            
            # Regular paragraph: split into sentences
            sentences = nltk.sent_tokenize(text)
            for i, sentence in enumerate(sentences):
                # Update progress for each sentence
                content_units_processed += 1
                percent_done = (content_units_processed / total_content_units) * 100
                eta = calculate_eta(start_time, content_units_processed, total_content_units)
                
                print(f"Translating sentence {i+1}/{len(sentences)} of paragraph {paragraph_number}...")
                
                try:
                    jp_translation = translate_text(sentence)
                    print(f"Done: {content_units_processed}/{total_content_units} ({percent_done:.1f}%) ETA: {eta} [{jp_translation}]")
                    
                    # Write each sentence immediately and flush to disk
                    save_to_file(jp_translation + "\n")
                except Exception as e:
                    print(f"Error translating sentence {i+1} of paragraph {paragraph_number}: {e}")
                    # Skip writing errors to output file for clean printing
            
            # Paragraph break
            save_to_file("\n")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Translation complete! Output written to {OUTPUT_FILE}")
    print(f"Statistics: {manuscript_title_count} manuscript title(s), {chapter_title_count} chapter title(s), {paragraph_number} paragraphs")
    print(f"Total content units: {total_content_units}")
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()
