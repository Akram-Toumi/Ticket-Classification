# Ticket Classification Model

This repository contains a PyTorch implementation of a ticket classification model using word embeddings and a 1D convolutional neural network (CNN). The project demonstrates end-to-end processing of text data for multiclass classification.

## Project Overview
The goal of this project is to classify tickets (e.g., support tickets, customer inquiries) into predefined categories based on their text content. The process includes:

1. Preprocessing the text data.
2. Padding/truncating sentences to a fixed length.
3. Splitting the data into training and test sets.
4. Defining and training a CNN-based model for classification.
5. Evaluating the model using accuracy, precision, and recall metrics.

## Installation

Ensure that you have Python 3.7+ installed, then install the required packages:

```bash
pip install torch torchmetrics scikit-learn nltk numpy pandas
```

## Dataset

The dataset includes:

- **words.json**: A JSON file containing a list of all unique words in the dataset.
- **text.json**: A JSON file with sentences represented as lists of words.
- **labels.npy**: A NumPy array containing the labels for each sentence.

These files should be placed in the same directory as the script.

## Usage

### Preprocessing
The script preprocesses the text by mapping words to indices and padding/truncating sentences to a fixed length of 50 tokens.

### Model Definition
The `TicketClassifier` class defines a CNN-based model with:

- An embedding layer for word representations.
- A 1D convolutional layer for feature extraction.
- A fully connected layer for classification.

### Training
The model is trained using the cross-entropy loss function and the Adam optimizer over 3 epochs. The batch size is set to 400.

### Evaluation
The script evaluates the model using accuracy, precision, and recall metrics, which are computed on the test dataset.

## Running the Script

To execute the script, ensure all dependencies are installed and dataset files are in place. Then, run:

```bash
python ticket_classifier.py
```

## Outputs

- **Training Loss**: Displays the average loss for each epoch.
- **Evaluation Metrics**:
  - Accuracy: Overall accuracy of the model.
  - Precision: Precision for each class.
  - Recall: Recall for each class.

### Example Output
```plaintext
Starting Training...
Epoch [1/3], Loss: 0.7854
Epoch [2/3], Loss: 0.5123
Epoch [3/3], Loss: 0.3847

Starting Evaluation...
Accuracy: 0.8574
Precision (per class): [0.89, 0.83, 0.88]
Recall (per class): [0.85, 0.84, 0.88]
```

## Model Parameters

- **Vocabulary Size**: Determined by the number of unique words in `words.json`.
- **Embedding Dimension**: 64
- **Sequence Length**: 50
- **Learning Rate**: 0.05
- **Number of Classes**: Determined by the unique labels in `labels.npy`.

## Dependencies

- Python 3.7+
- PyTorch
- TorchMetrics
- Scikit-learn
- NLTK
- NumPy
- Pandas

## Acknowledgments
This project was developed as an example of text classification using deep learning techniques.

