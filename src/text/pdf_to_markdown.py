from docling.document_converter import DocumentConverter

source = "./data/no5_20251106.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"