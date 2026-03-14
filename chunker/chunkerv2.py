from config import CHUNK_SIZE


def text_chunker(text, chunk_size=CHUNK_SIZE):
    """
    Line-based chunker with no overlap.

    1. Split on single newlines, keeping all non-empty lines.
    2. Lines are greedily grouped into line_chunks of up to chunk_size words.
       A new chunk starts whenever adding the next line would exceed the limit.

    Decision behind no overlap is so we don't have duplicates across line_chunks competing for top-k.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    line_chunks, chunk, chunk_len = [], [], 0
    for l in lines:
        words = l.split()
        if chunk_len + len(words) > chunk_size and chunk:
            line_chunks.append(" ".join(chunk))
            chunk, chunk_len = [], 0
        chunk.extend(words)
        chunk_len += len(words)
    if chunk:
        line_chunks.append(" ".join(chunk))

    return line_chunks
