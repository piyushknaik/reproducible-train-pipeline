import mlflow
import os
import time
import pandas as pd
from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.dataset_source import LocalArtifactDatasetSource

load_dotenv()

mlflow.set_experiment("genai-prompt-tracking-experiment")


llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def call_llm(prompt: str):
    """
    Actual LLM call using ChatGroq for experiment tracking.
    Returns the response, token count, and processing time.
    """
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Extract response text
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Estimate tokens: prompt tokens + response tokens (rough word-based estimate)
    prompt_tokens = len(prompt.split())
    response_tokens = len(response_text.split())
    total_tokens = prompt_tokens + response_tokens
    
    return response_text, total_tokens, time_taken

def test_prompt_variant(prompt: str, test_inputs: List[str], variant_name: str):
    with mlflow.start_run(run_name=variant_name):
        # Log the evaluation dataset
        eval_data = pd.DataFrame({"input": test_inputs})
        eval_data.to_csv("eval_data.csv", index=False)
        dataset = PandasDataset(eval_data, source=LocalArtifactDatasetSource("eval_data.csv"))
        mlflow.log_input(dataset, "evaluation_dataset")
        
        mlflow.log_param("prompt", prompt)
        
        total_tokens = 0
        total_time = 0
        details = []
        
        for inp in test_inputs:
            response, tokens, time_taken = call_llm(prompt.format(input=inp))
            total_tokens += tokens
            total_time += time_taken
            
            # Collect details for logging
            details.append({
                "input": inp,
                "response": response,
                "tokens": tokens,
                "time": time_taken
            })
        
        mlflow.log_metric("avg_tokens", total_tokens / len(test_inputs))
        mlflow.log_metric("avg_time", total_time / len(test_inputs))
        
        # Log detailed results as text artifact
        import json
        details_text = json.dumps(details, indent=2)
        with open("run_details.json", "w") as f:
            f.write(details_text)
        mlflow.log_artifact("run_details.json")

# Test different prompts
prompts = {
    "formal": "Please provide the capital city and the type of government for the following country: {input}",
    "casual": "Hey, what's the capital and government type of {input}?",
    "concise": "Capital and gov type of {input}:"
}

test_inputs = ["France", "Japan", "Brazil"]

if __name__ == "__main__":
    for name, prompt in prompts.items():
        test_prompt_variant(prompt, test_inputs, f"variant_{name}")