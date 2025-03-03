# adaptive-pipeline-config

**Brief Project Context:** 
This project is part of an adaptive, iterative model training pipeline that uses LLM suggestions to configure and reconfigure a Feed-Forward Network (FFN). The FFN is designed to achieve over 90% accuracy in predicting outcomes. The pipeline configures and trains the model, predicts and evaluates the results, and iteratively improves the model based on the performance of previous configurations.

**Purpose of this Repository:** 
The primary goal of this repository is to generate FFN model configurations based on suggestions from the Large Language Model (LLM) and device measurement data. The repository is responsible for creating, validating, and storing these configurations, as well as managing the pipeline's state and progress.

**Key Technologies:**
*   Python
*   JSON
*   Google Cloud Platform (GCP) Cloud Functions
*   TensorFlow
*   OpenAI API

**Getting Started**

**Prerequisites:**
*   Python 3.7 or higher
*   Google Cloud SDK (if deploying as functions)
*   OpenAI API key

**Installation Instructions:**
1. Clone the repository:
    ```bash
    git clone https://github.com/odegay/adaptive-pipeline-config.git
    cd adaptive-pipeline-config
    ```
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

**Usage/Examples**

*   **Inputs:**
    *   LLM suggestions in JSON format
    *   Device measurement data in JSON format

*   **Outputs:**
    *   JSON configuration file for the FFN model.

**Example Usage:**
1. Run the main script to start the pipeline:
    ```bash
    python main.py
    ```
2. The script will generate a new model configuration based on the LLM suggestions and device measurement data, validate the configuration, and store it in the pipeline's state.

**License**
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You are free to use, share, and adapt the material for non-commercial purposes, provided you give appropriate credit to the original creator.
