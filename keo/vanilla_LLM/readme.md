# Vanilla LLM Instructions for NER, CR, NEL & RE Tasks

This guide provides instructions on how to use vanilla LLMs to solve Named Entity Recognition (NER), Coreference Resolution (CR), Named Entity Linking (NEL), and Relation Extraction (RE) tasks using two different approaches: OpenAI GPT-4 and Ollama.

## OpenAI GPT-4 Instructions (`keo/vanilla_LLM/gpt4.py`)

### 1. API Key Setup

   - You need to fill in the OpenAI API key that Professor Jiang provided.
   - Simply replace `'Your_OpenAI_Key'` in the script with the key provided.

### 2. Running the Script

   - Once the key is set, run `gpt4.py` to solve NER, CR, NEL, and RE tasks.
   - The script includes evaluation against the gold standards from the dataset located at `OMIn_dataset/gold_standard/raw`.

## Ollama Instructions (`keo/vanilla_LLM/ollama.py`)

### 1. Install Ollama

   - First, install [Ollama](https://ollama.com/) on your local device.
   - Install a compatible LLM (such as `llama3.1-7B`).

### 2. Verify Installation

   - Run the command: 
     ```bash
     ollama run llama3.1
     ```
   - This command ensures that the LLM is working correctly on your device.

### 3. Serve Ollama Locally

   - Start the Ollama server by typing:
     ```bash
     ollama serve
     ```
   - Then, open [http://localhost:11434/](http://localhost:11434/) in your web browser.
   - If the page displays "Ollama is running", the server is set up correctly.

### 4. Running the Script

   - After setting up Ollama, run `ollama.py` to solve NER, CR, NEL, and RE tasks.
   - The script also evaluates the LLM’s performance against gold standards from the dataset located at `OMIn_dataset/gold_standard/raw`.

## Evaluation Notes

- The provided scripts evaluate the LLM's performance for each of the four tasks based on the gold standards.
- **NER and NEL Evaluations**: The evaluations for these tasks are likely correct, as the evaluation logic has been extensively tested.
- **CR and RE Evaluations**: The evaluations for these tasks may contain inaccuracies. Any assistance or suggestions to improve the evaluation metrics for Coreference Resolution and Relation Extraction are welcome.

Feel free to reach out if you encounter any issues or need further clarification.