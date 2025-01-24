import easyocr

def extract_text_from_plate(preprocessed_plate):
    reader = easyocr.Reader(['en'])  # Add languages as needed (e.g., 'es', 'zh')
    results = reader.readtext(preprocessed_plate, detail=0)
    return results[0] if results else 'None'
