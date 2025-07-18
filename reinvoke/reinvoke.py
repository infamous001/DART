import openai
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

file_path = '/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/doc_embed_dict.json'

client=openai.OpenAI()

with open(file_path, 'r') as file:
    data = json.load(file)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def extract_learning_intents(student_query):

    prompt = (
        "Instructions:\n"
        "Suppose you are a learning intent analyzer, and your task is to extract the underlying learning intents from the student's query.\n"
        "You should preserve all the learning goals expressed by the student, and the extracted learning intents should be easily understood without requiring additional context.\n"
        "Carefully read the given student query to understand its various learning intents.\n"
        "Then, identify the specific topics or concepts the student wants to learn.\n"
        "Each individual learning intent should be separated by a newline.\n\n"
        
        "Here are some examples of how you should solve the task.\n\n"

        "Example:\n"
        "Student Query: Can you help me learn about backpropagation and how neural networks are trained?\n"
        "Learning Intents:\n"
        "Learn about the concept of backpropagation.\n"
        "Understand the training process of neural networks.\n\n"

        "Student Query: Can you explain how quantum entanglement works and how it differs from classical physics?\n"
        "Learning Intents:\n"
        "Understand the concept of quantum entanglement.\n"
        "Learn how quantum entanglement differs from classical physics principles.\n\n"

        "Student Query: I need to learn about the symbolism used in Shakespeare’s Hamlet and how it reflects the themes of the play.\n"
        "Learning Intents:\n"
        "Study the symbolism used in Shakespeare’s Hamlet.\n"
        "Understand how symbolism reflects the themes of the play.\n\n"

        "Student Query: Can you help me understand the causes of World War I and how they led to the eventual conflict?\n"
        "Learning Intents:\n"
        "Learn about the causes of World War I.\n"
        "Understand the sequence of events that led to the conflict.\n\n"

        "Student Query: I want to know the differences between supervised, unsupervised, and reinforcement learning.\n"
        "Learning Intents:\n"
        "Understand supervised learning.\n"
        "Understand unsupervised learning.\n"
        "Learn about reinforcement learning and how it differs from other types of learning.\n\n"

        "Begin!\n"
        f"Student Query: {student_query}\n"
        "Learning Intents:\n"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    response_text = response.choices[0].message.content.strip()
    
    learning_intents = [intent.strip() for intent in response_text.split('\n') if intent.strip()]
    
    return learning_intents


def run_reinvoke(student_query):
 learning_intents = extract_learning_intents(student_query)

 intent_embed=[]
 for intent in learning_intents:
    embed = model.encode([intent])
    intent_embed.append(embed)


 intent_embed=np.array(intent_embed).squeeze(axis=1)
 query_embed = np.array(list(data.values())).squeeze(axis=1)
 sim_array = cosine_similarity(intent_embed, query_embed)



 ranked_array = np.argsort(np.argsort(sim_array, axis=1), axis=1) + 1

 ranked_with_score = np.array([
    [(ranked_array[i, j], sim_array[i, j],j) for j in range(sim_array.shape[1])]
    for i in range(sim_array.shape[0])
 ], dtype=object)

 num_columns = sim_array.shape[1]

 best_tuples = []

 count=0
 for col in range(num_columns):
    count=count+1
    col_tuples = [ranked_with_score[row][col] for row in range(sim_array.shape[0])]

    best_tuple = sorted(col_tuples, key=lambda x: (x[0], x[1],count))[-1]
    best_tuples.append(best_tuple)

 best_tuples_sorted = sorted(best_tuples, key=lambda x: (-x[0], -x[1]))

 doc_keys = list(data.keys())
 final_list=[]
 notes_path = "/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/notes"
 for item in best_tuples_sorted:
    file_name = doc_keys[item[2]]
    file_path = os.path.join(notes_path, file_name)
    final_list.append(file_path)
 return final_list



