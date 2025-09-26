# Causal-Based Anomaly Detection

This project implements an anomaly detection system that leverages causal discovery to identify deviations from normal behavior in network traffic data. By learning the underlying causal relationships between data features from benign data, the system can detect anomalies with a high degree of accuracy.

## Project Structure

The repository is organized as follows:

-   `raw_data/`: Directory to store raw CSV data files for processing.
-   `processed_data/`: Contains the cleaned and prepared data files.
-   `results/`: Stores the output of the analysis, including the learned causal graph and anomaly scores.
-   `check.py`: A utility script to verify the environment and CUDA installation.
-   `prepare_data.py`: Script for cleaning and preparing the raw data.
-   `learn_causal_graph.py`: Script to learn the causal graph from benign data.
-   `detect_anomalies.py`: Script that trains causal models and calculates anomaly scores for test data.

## Installation

To set up the environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have PyTorch installed with CUDA support if you have a compatible GPU.

## Usage

Follow these steps to run the anomaly detection pipeline:

1.  **Verify Environment (Optional):**
    Run `check.py` to ensure PyTorch and CUDA are correctly configured.
    ```bash
    python check.py
    ```

2.  **Prepare Data:**
    Place your raw CSV files in the `raw_data/` directory. Then, run the data preparation script.
    ```bash
    python prepare_data.py
    ```

3.  **Learn Causal Graph:**
    Execute the script to learn the causal graph from the prepared benign data.
    ```bash
    python learn_causal_graph.py
    ```

4.  **Detect Anomalies:**
    Finally, run the anomaly detection script to train the causal models and compute anomaly scores for the test data.
    ```bash
    python detect_anomalies.py
    ```

The final anomaly scores will be saved in `results/anomaly_scores.csv`.