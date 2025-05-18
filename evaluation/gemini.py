import os 
from google.genai import types, Client

def gemini_evaluation(image_bytes: bytes, eval_object: str, gemini_key) -> str:
    client = Client(api_key=gemini_key)

    prompt = (
        f"Please look at the image and determine if it contains {eval_object}. Respond with '1' if there is at least one {eval_object} visible, or '0' if there are none."
    )

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0
        )
    )
    res = response.text.strip() 
    if res == "1": 
        result = 1 
    elif res == "0": 
        result = 0 
    else:
        raise ValueError("Giá trị của 'res' không hợp lệ. Chỉ được nhận '1' hoặc '0'.")
    
    return result