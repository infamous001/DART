import os
import json
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from text_extraction import text_extract

client=openai.OpenAI()

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def generate_query(text,temp):
    task_prompt = (
        "Suppose you are an assistant and you have access to the following study material to answer user's queries. "
        "You are provided with a study material and its contents, including explanations, descriptions, and relevant information.\n\n"
        "Your task is to generate a possible user query that can be addressed by the topics explained in the chapter.\n"
        "You must include the input topics covered in the chapter. Please be creative and generate random but specific scenarios "
        "where the user would require explanations, clarifications, or guidance on the topics discussed.\n"
        "Now you are given the book chapter below:\n\n"
        f"{text}\n\n"
        "Please generate a user query that would require the knowledge provided in this chapter. "
        "Note that the generated query should be complex enough to describe realistic scenarios where a user would need assistance from the topics explained in the chapter.\n\n"
        "The relevant query is:"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": task_prompt}
        ],
        max_tokens=200,
        temperature=temp
    )
    
    return response.choices[0].message.content.strip()


def process_notes(notes_path):
    results = {}
    for filename in os.listdir(notes_path):
            file_path = os.path.join(notes_path, filename)
            text = text_extract(file_path)
            embed_list = []
    
            for i in range(10):
                temperature = 0.1 * (i + 1)
                query = generate_query(text, temperature)
                print(query)
                final_text = f"{text}\n\nquery:\n{query}"
    
                embed = model.encode([final_text])
                embed_list.append(embed)
    
            # Compute the mean embedding vector
            embed_mean = np.mean(embed_list, axis=0)
            results[file_path] = embed_mean.tolist()  # Convert numpy array to list for JSON compatibility

    return results

if __name__ == "__main__":
    notes_path = "/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/notes"
    output_file = "/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/doc_embed_dict1.json"

    embeddings = process_notes(notes_path)

    with open(output_file, 'w') as file:
        json.dump(embeddings, file, indent=4)

