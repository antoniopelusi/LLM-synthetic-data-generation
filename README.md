# LLM-synthetic-data-generation

The project follows a three-step process using the [pointe77/credit-card-transaction](https://huggingface.co/datasets/pointe77/credit-card-transaction) dataset to evaluate the quality of synthetic data across different configurations:

1. [**Dataset Analysis**](analysis/analysis.ipynb): Analyzing the original dataset to extract schema, distributions, and patterns. This information is used to engineer the system prompts required for effective generation.

2. [**Synthetic Data Generation**](generation/generation.ipynb): Running the generation process for each strategy applied to both [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) models:
    - Zero-shot In-Context Learning
    - Few-shot In-Context Learning (1 record)
    - Few-shot In-Context Learning (5 records)
    - Few-shot In-Context Learning (10 records)
    - LoRA Fine-Tuning (10% of the dataset)
    - LoRA Fine-Tuning (50% of the dataset)

3. [**Synthetic Data Comparison and Evaluation**](comparison/comparison.ipynb): Comparing synthetic results against the ground truth using statistical metrics and qualitative inspection.

## Setup & Run

Run the initial setup once using the following command:
```sh
make setup
```
This command will:
- remove any previous virtual environment (`.venv`)
- create a new virtual environment (`.venv`)
- install the requirements (`requirements.txt`)
- create the `.venv` kernel for Jupyter notebook

Activate the virtual environment prior to executing any Python scripts using the following command:
```sh
source .venv/bin/activate
```

If it's a Jupyter notebook, run it using the `.venv` kernel.
