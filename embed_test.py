from bedrock_llm import embed_text

v = embed_text("Hello world")

print(len(v))
print(v[:5])
