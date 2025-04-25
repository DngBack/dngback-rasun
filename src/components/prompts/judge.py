from __future__ import annotations

PROMPT_WITHOUT_CONTEXT = """You are an expert evaluator for the output of a large language model (LLM). Your job is to assess whether the generated response is appropriate and grounded, given the query and the context chunk from the source document.

Instructions:
Based on the provided context, determine whether the response:

    Accurately reflects information from the context chunk?

    Correctly and sufficiently answers the query?

    Avoids introducing information that is not in the context?

Output:
Respond with one of the following labels:

    YES – if the response is accurate, grounded in the context, and answers the query appropriately.

    NO – if the response is inaccurate, unrelated, or includes unsupported claims.

Input:

Query: {instruction}

LLM Response: {output}

Output: {label}
"""

PROMPT_WITH_CONTEXT = """You are an expert evaluator for the output of a large language model (LLM). Your job is to assess whether the generated response is appropriate and grounded, given the query and the context chunk from the source document.

Instructions:
Based on the provided context, determine whether the response:

    Accurately reflects information from the context chunk?

    Correctly and sufficiently answers the query?

    Avoids introducing information that is not in the context?

Output:
Respond with one of the following labels:

    YES – if the response is accurate, grounded in the context, and answers the query appropriately.

    NO – if the response is inaccurate, unrelated, or includes unsupported claims.

Input:

Query: {instruction}

Chunk: {input}

LLM Response: {output}

Output: {label}
"""
