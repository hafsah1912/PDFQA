import os
import re
import torch
import nltk
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer
import pdfplumber
import mysql.connector
import base64
import traceback
import unicodedata
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
asking_words = {
    'give', 'info', 'tell', 'show', 'find', 'list', 'describe', 'provide',
    'obtain', 'get', 'fetch', 'display', 'return', 'extract', 'identify',
    'explain', 'define', 'summarize', 'outline', 'discuss', 'analyze',
    'present', 'illustrate', 'detail', 'specify', 'mention', 'state',
    'indicate', 'confirm', 'verify', 'check', 'determine', 'ascertain',
    'require', 'need', 'want', 'seek', 'look'
}
question_starters = {
    'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'how',
    'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'can', 'could', 'shall', 'should', 'will', 'would',
    'may', 'might', 'must'
}
misc_common = {
    'about', 'on', 'in', 'for', 'of', 'with', 'by', 'from', 'at', 'to',
    'and', 'or', 'but', 'if', 'then', 'else', 'than', 'that', 'this', 'these', 'those',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'her', 'it', 'us', 'them',
    'a', 'an', 'the', 'some', 'any', 'no', 'each', 'every', 'own', 'other', 'such',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'only', 'just', 'even', 'very', 'so', 'too', 'enough', 'rather', 'quite',
    'across', 'after', 'against', 'among', 'around', 'before', 'behind', 'below',
    'beneath', 'beside', 'between', 'beyond', 'during', 'except', 'inside', 'into',
    'near', 'off', 'out', 'outside', 'over', 'past', 'through', 'under', 'up', 'upon',
    'while', 'whereupon', 'whereas', 'down', 'since', 'until', 'towards', 'within',
    'without', 'throughout', 'below', 'above', 'once', 'here', 'there', 'whence',
    'wherever', 'whenever', 'whichever', 'whoever', 'whomever', 'whatever'
}
stop_words.update(asking_words)
stop_words.update(question_starters)
stop_words.update(misc_common)
format_indicator_words = {
    'full', 'para', 'paragraph', 'section', 'details', 'information',
    'summary', 'overview', 'report', 'list', 'steps', 'procedure',
    'guide', 'description', 'specification', 'overview'
}
class FileEmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"✅ Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
    def generate_embedding(self, text: str) -> Union[torch.Tensor, None]:
        if self.model is None or self.tokenizer is None:
            print("⚠️ Embedding model not loaded. Cannot generate embedding.")
            return None
        if not isinstance(text, str) or not text.strip():
             return torch.zeros(self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 384) # .to(self.device)
        try:
            max_len = self.tokenizer.model_max_length
            if max_len is None or max_len > 10000: # Use a reasonable upper cap
                 max_len = 512
            encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            with torch.no_grad():
                output = self.model(**encoding)
            input_mask_expanded = encoding['attention_mask'].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(output.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings.squeeze().detach()
        except Exception as e:
            print(f"❌ Error generating embedding for text: '{text[:100]}...'. Error: {e}")
            return None
    def cosine_similarity(self, emb1: Union[torch.Tensor, None], emb2: Union[torch.Tensor, None]) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        emb1 = emb1.view(1, -1) if emb1.ndim == 1 else emb1
        emb2 = emb2.view(1, -1) if emb2.ndim == 1 else emb2
        if torch.norm(emb1) == 0 or torch.norm(emb2) == 0:
            return 0.0
        try:
            return F.cosine_similarity(emb1, emb2).item()
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return 0.0
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="file_embeddings_db"
    )
def save_embeddings(embeddings: List[torch.Tensor], texts: List[str]):
    db = connect_to_db()
    cursor = db.cursor()
    insert_query = "INSERT INTO embeddings (sentence, embedding) VALUES (%s, %s)"
    for text, emb in zip(texts, embeddings):
        emb_bytes = base64.b64encode(emb.numpy().tobytes()).decode("utf-8")
        cursor.execute(insert_query,(text, emb_bytes))
    db.commit()
    cursor.close()
    db.close()
    print("✅ Embeddings saved to MySQL database.")
def load_embeddings() -> Tuple[List[torch.Tensor], List[str]]:
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("SELECT sentence, embedding FROM embeddings")
    texts = []
    embeddings = []
    for text, emb_str in cursor.fetchall():
        emb_bytes = base64.b64decode(emb_str)
        tensor = torch.tensor(np.frombuffer(emb_bytes, dtype=np.float32))
        texts.append(text)
        embeddings.append(tensor)
    cursor.close()
    db.close()
    return embeddings, texts

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reliable PDF text extraction with bullet support via pdfplumber."""
    try:
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Use layout-based extraction (sometimes bullets are in different columns)
                lines = page.extract_text(layout=True)
                if lines:
                    full_text.append(lines)
        return "\n".join(full_text)
    except Exception as e:
        print(f"❌ Error extracting text from {os.path.basename(pdf_path)} using pdfplumber: {e}")
        return ""
def normalize_text(text: str) -> str:
    """
    Basic text normalization including Unicode normalization,
    removing non-ASCII, and standardizing whitespace.
    Also handles some common PDF parsing issues.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters
    # Allow basic punctuation, alphanumeric, and common symbols in documents
    text = re.sub(r'[^a-zA-Z0-9\s:.,;\'"?!@#$%^&*()_+-=\[\]{}|<>\/]', '', text) # Broaden allowed characters
    text = re.sub(r'\s+', ' ', text).strip() # Standardize whitespace
    text = re.sub(r'\(?cid:\d+\)?', '', text) # Remove cid artifacts
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text) # Remove repeated words
    return text
def clean_garbage(text: str) -> str:
    """Removes common PDF extraction artifacts and repeated words."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\(?cid:\d+\)?', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return text.strip()
def merge_cross_page_rows(row1: str, row2: str) -> str:
    """Merges two strings, intended for table rows split across pages."""
    if not isinstance(row1, str): row1 = ""
    if not isinstance(row2, str): row2 = ""
    row1_stripped = row1.strip()
    if row1_stripped.endswith(":"):
         row1_stripped = row1_stripped[:-1]
    return row1_stripped + " " + row2.strip()
def extract_pdf_table(pdf_path: str) -> List[str]:
    """Extracts tables from a PDF and returns rows as strings."""
    rows = []
    prev_row_buffer = None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    df = pd.DataFrame(table).fillna("").astype(str)
                    for idx, (_, row) in enumerate(df.iterrows()):
                        current_row_str_parts = (normalize_text(str(cell).strip()) for cell in row if str(cell).strip())
                        current_row_str = " ".join(current_row_str_parts)
                        if not current_row_str.strip():
                            continue
                        if prev_row_buffer:
                            merged_row = merge_cross_page_rows(prev_row_buffer, current_row_str)
                            rows.append(clean_garbage(merged_row))
                            prev_row_buffer = None # Reset buffer after merging
                        else:
                            if idx == len(df) - 1 and page_index < len(pdf.pages) - 1:
                                prev_row_buffer = current_row_str
                            else:
                                rows.append(clean_garbage(current_row_str))
    except Exception as e:
        print(f"⚠️ Error extracting table from {os.path.basename(pdf_path)}: {e}")
    return rows


def group_paragraphs_with_structure(text: str) -> List[str]:
    """
    Groups lines of text into paragraphs, trying to preserve headings and list structures.
    More advanced grouping than the previous version.
    """
    if not isinstance(text, str):
        return []

    lines = text.splitlines()
    paragraphs = []
    current_para_lines = []

    # Patterns for different types of lines
    heading_pattern = re.compile(r"^[A-Z][a-zA-Z0-9\s,.'\"&]*:?\s*$") # More flexible heading pattern
    bullet_pattern = re.compile(r"^\s*[-•*o]|[ivxlcdmIVXLCDM]+\s*[.)]|\([ivxlcdmIVXLCDM]+\)|[a-zA-Z]\s*[.)]|\([a-zA-Z]\)\s+.*") # More comprehensive bullet pattern
    numbered_list_pattern = re.compile(r"^\s*\d+[.)]\s+.*") # Numbered list pattern

    def finalize_current_paragraph():
        """Joins current lines and adds to paragraphs list if not empty."""
        if current_para_lines:
            paragraph = " ".join(current_para_lines).strip()
            if paragraph: # Ensure paragraph is not empty after joining
                 paragraphs.append(paragraph)
            current_para_lines.clear() # Use .clear() for efficiency

    for i, line in enumerate(lines):
        clean_line = normalize_text(line.strip())
        if not clean_line:
            # Treat blank lines as potential paragraph separators, unless within a list context
            if not current_para_lines or (not bullet_pattern.match(current_para_lines[-1]) and not numbered_list_pattern.match(current_para_lines[-1])):
                finalize_current_paragraph()
            continue # Skip the empty line itself

        is_bullet = bool(bullet_pattern.match(clean_line))
        is_numbered = bool(numbered_list_pattern.match(clean_line))
        is_heading = bool(heading_pattern.match(clean_line)) # and len(clean_line.split()) < 10 # Option to restrict heading length


        start_new = False
        if is_heading:
             start_new = True
        elif (is_bullet or is_numbered):
            if current_para_lines:
                 last_line = normalize_text(current_para_lines[-1]).strip()
                 # Start new if last line was NOT a bullet/numbered item and NOT a heading
                 if not bullet_pattern.match(last_line) and not numbered_list_pattern.match(last_line) and not heading_pattern.match(last_line):
                      start_new = True
            else:
                 # If current_para_lines is empty, this is the first line or after a blank line
                 start_new = True # Always start a new paragraph with a bullet/numbered item if the buffer is empty


        # If starting a new paragraph, finalize the previous one
        if start_new:
            finalize_current_paragraph()

        current_para_lines.append(clean_line)

    # Add the last accumulated paragraph
    finalize_current_paragraph()

    return [p for p in paragraphs if p] # Return only non-empty paragraphs
def merge_split_keys(paragraphs: List[str]) -> List[str]:
    merged = []
    skip_next = False
    for i in range(len(paragraphs)):
        if skip_next:
            skip_next = False
            continue
        current_para = paragraphs[i]
        if not isinstance(current_para, str) or not current_para.strip():
             merged.append(current_para) # Append as is (likely None or empty) and continue
             continue
        if re.search(r':\s*$', current_para) and len(current_para.split()) > 1 and i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1]
            if isinstance(next_para, str):
                 merged_para = current_para + " " + next_para
                 merged.append(merged_para)
                 skip_next = True
            else:
                 merged.append(current_para)
        else:
            merged.append(current_para)
    return [p for p in merged if isinstance(p, str) and p.strip()]
def extract_keywords(text: str) -> List[str]:
    """Extracts relevant non-stopword keywords."""
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[\w\-]+\b', text.lower())
    return [w for w in words if w not in stop_words]
def preprocess_question(question: str) -> str:
    if not isinstance(question, str):
        return ""
    normalized = normalize_text(question)
    return normalized.lower()
def keyword_overlap_score(q: str, s: str) -> float:
    if not isinstance(q, str) or not isinstance(s, str):
        return 0.0
    q_lower = q.lower()
    s_lower = s.lower()
    vectorizer = CountVectorizer(token_pattern=r'\b[\w\-]+\b').fit([q_lower, s_lower])
    vectors = vectorizer.transform([q_lower, s_lower]).toarray()
    if vectors.sum() == 0:
        return 0.0
    intersection = np.minimum(vectors[0], vectors[1])
    question_sum = np.sum(vectors[0])
    score = np.sum(intersection) / (question_sum + 1e-5)
    if q_lower in s_lower:
        score += 0.2
    return score
def strict_keyword_match(query: str, sentence: str) -> bool:
    if not isinstance(query, str) or not isinstance(sentence, str):
        return False
    all_query_keywords = set(extract_keywords(query))
    core_subject_tokens = {
        token for token in all_query_keywords
        if token not in format_indicator_words
    }
    if not core_subject_tokens:
        return True
    sentence_lower = sentence.lower()
    sentence_words = set(re.findall(r'\b[\w\-]+\b', sentence_lower))
    return len(core_subject_tokens.intersection(sentence_words)) > 0

def interactive_qa(generator: FileEmbeddingGenerator, embeddings: List[Union[torch.Tensor, None]], paragraphs: List[str], similarity_threshold=0.3): # Reverted threshold to a value typically between 0 and 1
    """
    Handles interactive question answering based on document embeddings.
    Includes improved logic to group headings with subsequent bullet points in the output.
    Refined scoring and result presentation.
    """
    if not paragraphs or not embeddings or len(paragraphs) != len(embeddings):
        print("\n🚫 Searchable content or embeddings not loaded correctly or lists are misaligned. Cannot start Q&A.")
        return

    print("\n💬 Ask questions based on the PDF content (type 'exit' to quit):\n")

    # Define patterns for identifying headings and list items (consistent with group_paragraphs_with_structure)
    heading_pattern = re.compile(r"^[A-Z][a-zA-Z0-9\s,.'\"&]*:?\s*$")
    bullet_pattern = re.compile(r"^\s*[-•*o]|[ivxlcdmIVXLCDM]+\s*[.)]|\([ivxlcdmIVXLCDM]+\)|[a-zA-Z]\s*[.)]|\([a-zA-Z]\)\s+.*")
    numbered_list_pattern = re.compile(r"^\s*\d+[.)]\s+.*")


    while True:
        try:
            question = input("🟡 Your question: ").strip()
            if question.lower() == "exit":
                print("👋 Exiting Q&A.")
                break
            if not question:
                continue

            processed_q = preprocess_question(question)
            question_keywords = extract_keywords(processed_q)
            core_subject_keywords = {
                kw for kw in question_keywords if kw not in format_indicator_words
            }

            if not processed_q or not core_subject_keywords:
                print("❌ Question contains no recognizable subject keywords. Try a more specific query.")
                continue

            q_embedding = generator.generate_embedding(processed_q)
            if q_embedding is None:
                print("❌ Could not generate embedding for your question.")
                continue

            scored = []
            for i, (para, emb) in enumerate(zip(paragraphs, embeddings)): # Added index 'i'
                if not isinstance(para, str) or not para.strip() or emb is None:
                    continue

                norm_para = normalize_text(para)
                norm_para_lower = norm_para.lower()

                # Initialize scores to 0.0 at the start of each iteration
                sim = 0.0
                substring_bonus = 0.0
                kw_score = 0.0
                list_structure_bonus = 0.0
                key_value_bonus = 0.0


                # Strict keyword match filter - ensure at least one core keyword is present
                if not strict_keyword_match(processed_q, norm_para):
                    continue # Skip if no core keywords match

                # Calculate scores *after* the strict keyword match filter
                sim = generator.cosine_similarity(q_embedding, emb)

                # Prioritize exact phrase match of processed question
                substring_bonus = 0.6 if processed_q in norm_para_lower else 0

                # Calculate keyword overlap score
                kw_score = keyword_overlap_score(processed_q, norm_para)


                # Bonus for lines that match list item or heading patterns
                list_structure_bonus = 0.3 if (
                    bullet_pattern.match(norm_para) or
                    numbered_list_pattern.match(norm_para) or
                    heading_pattern.match(norm_para)
                ) else 0


                # Apply a bonus if the paragraph contains a colon and matches a core keyword before it
                if ':' in norm_para:
                    try:
                        key, val = norm_para.split(':', 1)
                        if any(kw in key.lower() for kw in core_subject_keywords):
                            key_value_bonus += 0.4
                        # Add a small bonus for shorter key-value like entries
                        if len(norm_para.split()) <= 20:
                            key_value_bonus += 0.2
                    except Exception:
                        pass # Ignore errors during key-value parsing attempts


                # Combine scores - adjusted weights back to values typically less than 1
                final_score = (
                    sim * 0.7 +             # Cosine similarity remains a key factor
                    kw_score * 0.4 +        # Keyword overlap
                    key_value_bonus * 0.3 + # Key-value structure bonus (bonus * weight)
                    list_structure_bonus * 0.3 + # List/Heading structure bonus (bonus * weight)
                    substring_bonus         # Exact substring match bonus
                )

                # Add the paragraph and its score/original index if it meets the threshold
                if final_score >= similarity_threshold:
                    scored.append((final_score, para, i)) # Store score, text, and original index

            # Remove potential duplicate paragraphs while keeping the highest score
            unique_scored = {}
            for score, para, original_index in scored:
                if para not in unique_scored or score > unique_scored[para][0]:
                    unique_scored[para] = (score, para, original_index)

            # Get the original indices of all paragraphs that passed the initial filter
            relevant_indices = {item[2] for item in unique_scored.values()}


            # Sort results by score in descending order (for the max score calculation)
            sorted_scored = sorted(unique_scored.values(), key=lambda x: x[0], reverse=True)

            # Find the actual maximum score among the retrieved results for scaling
            actual_max_score = sorted_scored[0][0] if sorted_scored else 0.0

            # --- Refined Logic to Aggregate Related Relevant Items Based on Original Document Order ---
            output_paragraphs = []
            added_to_output = set() # Track indices from the original paragraphs list that have been added

            # Iterate through paragraphs in their original document order
            for i, para in enumerate(paragraphs):
                 # Only consider paragraphs that were identified as relevant initially
                 if i not in relevant_indices:
                     continue

                 # If this relevant paragraph hasn't been added yet (as part of a previous aggregation block)
                 if i not in added_to_output:
                     output_paragraphs.append(para)
                     added_to_output.add(i)

                     norm_para = normalize_text(para)
                     is_heading = bool(heading_pattern.match(norm_para))
                     is_bullet = bool(bullet_pattern.match(norm_para))
                     is_numbered = bool(numbered_list_pattern.match(norm_para))

                     # If this relevant paragraph is a heading or a list item,
                     # greedily add subsequent *relevant* list items or lines.
                     if is_heading or is_bullet or is_numbered:
                          for j in range(i + 1, len(paragraphs)):
                               next_para = paragraphs[j]
                               # ONLY add the next paragraph if it's also in the relevant_indices
                               if j in relevant_indices:
                                    output_paragraphs.append(next_para)
                                    added_to_output.add(j)
                                    norm_next_para = normalize_text(next_para)
                                    is_next_bullet = bool(bullet_pattern.match(norm_next_para))
                                    is_next_numbered = bool(numbered_list_pattern.match(norm_next_para))
                                    is_next_heading = bool(heading_pattern.match(norm_next_para))

                                    # Stop aggregating this block if the next relevant item is a new heading,
                                    # or if the structure clearly changes.
                                    # This part might need tuning based on document structure.
                                    if is_next_heading: # Stop at the next relevant heading
                                         break
                                    # Also stop if the immediately following relevant item doesn't seem to
                                    # continue the list structure, assuming the current one was a list item.
                                    if (is_bullet or is_numbered) and not (is_next_bullet or is_next_numbered) and not is_next_heading:
                                        # This is a heuristic: if we were in a list and the next relevant item isn't
                                        # a list item or heading, maybe the block is finished.
                                        # Consider carefully if this heuristic fits your documents.
                                        pass # Let's not break here yet, just adding relevant items in sequence


                               else:
                                    # If the next paragraph is *not* in relevant_indices, stop this aggregation block.
                                    break


            # --- End Refined Aggregation Logic ---


            # Display results only if there are output paragraphs
            if output_paragraphs:
                print(f"\n✅ Found relevant entr{'y' if len(output_paragraphs) == 1 else 'ies'}:\n")
                score_lookup = {para: score for score, para, _ in unique_scored.values()}

                # Print the aggregated and unique results in the order they appeared in the original document
                for idx, ans in enumerate(output_paragraphs[:15], 1): # Limit the number of displayed paragraphs
                    trimmed = ans.strip()
                    if len(trimmed) > 1000: # Truncate very long outputs
                        trimmed = trimmed[:997] + "..."

                    # Get the raw score for the current paragraph
                    current_para_raw_score = score_lookup.get(ans, 0.0)

                    # Scale the score to a percentage between 0 and 100
                    # Handle division by zero if actual_max_score is 0
                    if actual_max_score > 0:
                         current_para_percentage = (current_para_raw_score / actual_max_score) * 100
                         # Ensure percentage is not more than 100 (due to floating point inaccuracies or edge cases)
                         current_para_percentage = min(current_para_percentage, 100.0)
                         # Ensure percentage is at least 1 (as requested by user)
                         current_para_percentage = max(current_para_percentage, 1.0)
                    else:
                         current_para_percentage = 0.0 # If max score is 0, all scores are 0


                    # Format the output to show percentage
                    print(f"\n--- Result {idx} (Accuracy: {current_para_percentage:.2f}%) ---")
                    print(trimmed)
                    print("-" * 40)

            else:
                # If no paragraphs made it to the output_paragraphs list after aggregation/filtering
                print("❌ No relevant answer found based on criteria.")
                print(f"💡 Consider rephrasing or checking if relevant keywords appear in the text.")


        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Exiting Q&A.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            traceback.print_exc()
def main(pdf_paths: List[str]):
    """Main function to process PDFs, generate/load embeddings, and start Q&A."""
    all_processed_entries = []
    print("📄 Starting PDF content extraction and processing...")
    for pdf_path in pdf_paths:
        if not os.path.isfile(pdf_path):
            print(f"❌ Invalid file path provided: {pdf_path}")
            continue
        print(f"⏳ Processing: {os.path.basename(pdf_path)}...")
        pdf_text = extract_text_from_pdf(pdf_path)
        table_rows = extract_pdf_table(pdf_path)
        text_paragraphs = group_paragraphs_with_structure(pdf_text)
        text_paragraphs = merge_split_keys(text_paragraphs)
        combined_entries = [entry for entry in text_paragraphs + table_rows if isinstance(entry, str) and entry.strip()]
        cleaned_entries = [clean_garbage(normalize_text(s)) for s in combined_entries if isinstance(s, str) and s.strip()]
        all_processed_entries.extend(cleaned_entries)
        print(f"✅ Finished processing: {os.path.basename(pdf_path)}. Grouped {len(text_paragraphs)} text entries and extracted {len(table_rows)} table rows.")
    unique_processed_entries = list(dict.fromkeys(entry for entry in all_processed_entries if isinstance(entry, str) and entry.strip()))
    print(f"\n🧠 Total unique searchable entries across all PDFs after initial processing: {len(unique_processed_entries)}")
    if not unique_processed_entries:
        print("🚫 No valid content extracted from PDFs. Cannot proceed with embedding or Q&A.")
        return
    generator = FileEmbeddingGenerator()
    if generator.model is None:
        print("🚫 Embedding model failed to load. Cannot proceed.")
        return
    loaded_embeddings, loaded_sentences = load_embeddings()
    current_embeddings = []
    current_sentences = []
    if loaded_sentences and len(loaded_embeddings) == len(loaded_sentences):
        print("✅ Using embeddings and sentences loaded from MySQL.")
        current_sentences = loaded_sentences
        current_embeddings = loaded_embeddings
        loaded_sentence_set = set(loaded_sentences)
        sentences_to_generate = [s for s in unique_processed_entries if s not in loaded_sentence_set]
        if sentences_to_generate:
            print(f"⚠️ Found {len(sentences_to_generate)} new unique entries not in DB. Generating embeddings for them...")
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                new_embeddings = list(executor.map(generator.generate_embedding, sentences_to_generate))
            valid_new_entries = [(sent, emb) for sent, emb in zip(sentences_to_generate, new_embeddings) if isinstance(sent, str) and sent.strip() and emb is not None]
            if valid_new_entries:
                new_sentences_filtered, new_embeddings_filtered = zip(*valid_new_entries)
                current_sentences.extend(new_sentences_filtered)
                current_embeddings.extend(new_embeddings_filtered)
                print(f"✅ Generated and added embeddings for {len(valid_new_entries)} new entries.")
                save_embeddings(new_embeddings_filtered, new_sentences_filtered)
            else:
                print("⚠️ No valid new embeddings were generated.")
    else:
        print("🔍 No valid embeddings loaded from MySQL or data is inconsistent. Generating embeddings for all unique entries...")
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            embeddings = list(executor.map(generator.generate_embedding, unique_processed_entries))
        valid_entries = [(sent, emb) for sent, emb in zip(unique_processed_entries, embeddings) if isinstance(sent, str) and sent.strip() and emb is not None]
        if not valid_entries:
            print("🚫 Failed to generate valid embeddings for any entries. Cannot proceed.")
            return
        current_sentences, current_embeddings = zip(*valid_entries)
        current_sentences = list(current_sentences)
        current_embeddings = list(current_embeddings)
        print(f"✅ Successfully generated embeddings for {len(current_sentences)} valid entries.")
        save_embeddings(current_embeddings, current_sentences)
    final_sentences = []
    final_embeddings = []
    seen_sentences = set()
    for sent, emb in zip(current_sentences, current_embeddings):
        if isinstance(sent, str) and sent.strip() and sent not in seen_sentences and emb is not None:
            final_sentences.append(sent)
            final_embeddings.append(emb)
            seen_sentences.add(sent)
    if len(final_sentences) == 0 or len(final_sentences) != len(final_embeddings):
        print("🚫 Final processed data or embeddings list is empty or misaligned after filtering. Cannot start Q&A.")
        return
    print(f"\n✅ Ready for Q&A with {len(final_sentences)} searchable entries.")
    interactive_qa(generator, final_embeddings, final_sentences)

if __name__ == "__main__":
    pdf_paths = [
        r"C:\Users\HI-TECH\Documents\PDF\BHEL LOA 04.10.2023.pdf",
        r"C:\Users\HI-TECH\Documents\PDF\72344023_TD_711_mb1G-1.pdf"
    ]
    main(pdf_paths)