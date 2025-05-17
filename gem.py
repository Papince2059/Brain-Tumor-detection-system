import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
key = os.getenv("API_KEY")  # Make sure your .env file contains this key

# Print the API key for debugging (Make sure it’s not empty)
print(f"Using API key: {key}")

# Configure the API with the API key
genai.configure(api_key=key)

def fetch_definition_data(disease_name):
    # Define your query with a focus on brevity and actionability for farmers
    query = f"Provide a brief summary of the cure and precautionary measures for {disease_name}. Focus on actionable steps and essential information that can be quickly understood by Patient in point wise."
    
    try:
        # Request content from the Gemini model
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            contents=[{"role": "user", "parts": [{"text": query}]}]
        )

        # Log the response for debugging purposes
        print(f"API Response: {response}")
        
        # Check if response is as expected (i.e., contains candidates and relevant text)
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            # Extract the generated text
            generated_text = response.candidates[0].content.parts[0].text
            return generated_text
        else:
            return "No suggestions found for this disease."
    
    except Exception as e:
        # Print detailed error message for debugging
        print(f"Error fetching suggestions: {e}")
        return f"Error fetching suggestions: {e}. Please try again later."
