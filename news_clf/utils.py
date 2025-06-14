import re

def shorten_to_n_words(text, n=1500):
    words = re.findall(r'\b\w+\b', text)
    if len(words) <= n:
        return text  # no truncation needed
    
    # Find the index where the n-th word ends
    count = 0
    end_index = len(text)
    for match in re.finditer(r'\b\w+\b', text):
        count += 1
        if count == n:
            end_index = match.end()
            break
    
    return text[:end_index].rstrip() + "[truncated]..."

def format_prompt_with_article(title, body, max_words=2000):
    body = shorten_to_n_words(body, n=max_words)
    article_input = f'Title: {title}[SEP]{body}'
    return article_input

def format_prompt_from_row(row, max_words=2000):
    return format_prompt_with_article(row.title, row.body, max_words=max_words)

