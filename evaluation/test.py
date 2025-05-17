from gemini import gemini_evaluation
import os 
from dotenv import load_dotenv
# load api 
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") 

with open('1.png', 'rb') as f: 
  data = f.read() 

print(gemini_evaluation(data, 'crab', gemini_key))

