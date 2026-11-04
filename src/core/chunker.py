import re

pattern_articles = r'(ARTICLE\s+[IVXLCDM]+|ARTICLE\s+\d+|Chapter\s+\d+)'
pattern_sections = r'(Section\s+\d+\.\d+|\d+\.\d+\.\s+|\d+\.\s+[A-Z][A-Z\s]*)'
pattern_sub_sections = r'(\d+\.\d+\.\d+\s+)'
pattern_annexes = r'(Annex\s+\d+|Exhibit\s+[A-Z]|Schedule\s+\d+\.\d+[\(a-z\)]*)'
pattern_recitals = r'(RECITALS|WHEREAS)'
pattern_letters = r'(\([a-z]+\)|\([i]+\))'

def clean_document(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*\d+\s*$', line.strip()):
            continue
        if line.strip().startswith("Source:"):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def split_by_pattern(text, pattern):
    parts = re.split(f'({pattern})', text, flags=re.IGNORECASE)
    chunks = []
    current_header = None
    current_body = []
    for part in parts:
        if re.match(pattern, part, re.IGNORECASE):
            if current_header is not None:
                full_text = ''.join(current_body).strip()
                if full_text:
                    chunks.append((current_header, full_text))
            current_header = part.strip()
            current_body = []
        else:
            current_body.append(part)
    if current_header is not None:
        full_text = ''.join(current_body).strip()
        if full_text:
            chunks.append((current_header, full_text))
    return chunks

def cut_long_chunk(header, text, max_size=2000, soft_size=1000):
    if len(text) <= max_size:
        return [(header, text)]
    sentences = text.split('. ')
    sub_chunks = []
    current_text = ""
    for sentence in sentences:
        if len(current_text) + len(sentence) > soft_size and current_text:
            sub_chunks.append(current_text.strip())
            current_text = sentence + ". "
        else:
            current_text += sentence + ". "
    if current_text:
        sub_chunks.append(current_text.strip())
    final_chunks = []
    total = len(sub_chunks)
    for i, chunk_text in enumerate(sub_chunks, 1):
        new_header = f"{header} (suite {i}/{total})"
        final_chunks.append((new_header, chunk_text))
    return final_chunks

def chunk_contract(full_text):
    text = clean_document(full_text)
    pattern_top_level = r'(ARTICLE\s+[IVXLCDM]+|ARTICLE\s+\d+|Chapter\s+\d+|Annex\s+\d+|Schedule\s+\d+\.\d+[\(a-z\)]*|\n\d+\.\s+[A-Z][A-Z\s]*)'
    first_match = re.search(pattern_top_level, text, re.IGNORECASE)
    preamble_text = ""
    remaining_text = text
    if first_match:
        start_pos = first_match.start()
        preamble_text = text[:start_pos].strip()
        remaining_text = text[start_pos:]
    final_chunks = []
    if preamble_text:
        final_chunks.append(("PREAMBLE", preamble_text))
    top_chunks = split_by_pattern(remaining_text, pattern_top_level)
    if not top_chunks:
        if not preamble_text:
            final_chunks = [("FULL_CONTRACT", text)]
        else:
            if not final_chunks:
                final_chunks = [("FULL_CONTRACT", text)]
    else:
        for top_header, top_body in top_chunks:
            section_chunks = split_by_pattern(top_body, pattern_sections)
            if not section_chunks:
                final_chunks.extend(cut_long_chunk(top_header, top_body))
            else:
                for sec_header, sec_body in section_chunks:
                    final_chunks.extend(cut_long_chunk(sec_header, sec_body))
    if not final_chunks:
        final_chunks = [("FULL_CONTRACT", text)]
    return final_chunks