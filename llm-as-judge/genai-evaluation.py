import os
import mlflow
from langchain_groq import ChatGroq
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from dotenv import load_dotenv
from openai import OpenAI
from mlflow.metrics.genai import answer_correctness

load_dotenv()

mlflow.set_experiment("GenAI Evaluation")

client = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)



# Your agent implementation
def my_agent(question: str) -> str:
    response = client.invoke([
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions concisely.",
        },
        {"role": "user", "content": question},
    ])
    return response.content


# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
    return my_agent(question)


# Evaluation dataset
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]


# Scorers
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 5

@scorer
def groq_correctness(inputs, outputs, expectations):
    question = inputs['question']
    expected = expectations['expected_response']
    prompt = f"Question: {question}\nExpected Answer: {expected}\nActual Answer: {outputs}\nIs the actual answer correct? Respond with 'yes' or 'no'."
    response = client.invoke([{"role": "user", "content": prompt}])
    return response.content.lower().strip() == 'yes'


scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
    groq_correctness
]

# Run evaluation
if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )