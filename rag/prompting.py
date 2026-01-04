from typing import List, Dict

def build_rag_prompt(question: str, matches: List[Dict[str, object]]) -> str:
    """
    Turns retrieved chunks into a single prompt that instructs the LLM
    to answer using only the provided context.
    """
    context_blocks = []
    for i, m in enumerate(matches, start=1):
        source = m["source"]
        chunk_id = m["chunk_id"]
        text = m["text"]
        context_blocks.append(f"[{i}] Source: {source} (chunk {chunk_id})\n{text}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the context does not contain the answer, say: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt
