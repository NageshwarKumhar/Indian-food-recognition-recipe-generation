# Indian Food Recognition and Recipe Generation

A Streamlit prototype for an MLLP assignment. It uses a pretrained Hugging Face image classifier and FLAN-T5-small to demonstrate an image-to-food-information pipeline.

## Features
- Upload a JPG, JPEG, PNG or WEBP food image.
- Display top image-classification predictions and scores.
- Show a curated regional association, description and typical recipe for supported foods.
- Generate a short text response using FLAN-T5-small.
- Correct the predicted dish manually for foods not covered by the classifier.

## Models
- Image classifier: [Subhash5/indian-food-classifier](https://huggingface.co/Subhash5/indian-food-classifier)
- Text generation: [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)

## Important limitation
The pretrained classifier supports a fixed set of 20 categories, including idli, dhokla and masala dosa. It does not directly classify every target dish. Dalma and Chhena Poda are included in the recipe knowledge base for demonstration, but are not automatically recognized by this classifier. To claim automatic recognition for those dishes, collect a labelled dataset and fine-tune a classifier with those classes.

Regional associations are curated knowledge, not inferred solely from pixels. Generated recipes should be reviewed.

## Setup on Windows
Recommended: Python 3.11, VS Code, and a terminal.

```powershell
# 1. Open this folder in VS Code, then run:
py -3.11 -m venv .venv

# 2. Activate the environment:
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use Command Prompt:
# .venv\Scripts\activate.bat

# 3. Install dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the app:
streamlit run app.py
```

The first run downloads the model files from Hugging Face and may take several minutes. An internet connection is required for the first download.

## Suggested evaluation
- Make a test set with images not used during model training.
- Record the expected class and predicted class for each image.
- Report accuracy, precision, recall and F1-score only when you have a labelled test set.
- Include examples of correct predictions, incorrect predictions and unsupported foods.

## Suggested repository structure
```text
indian-food-recognition/
├── app.py
├── requirements.txt
├── README.md
├── report/
│   └── project_report.pdf
├── presentation/
│   ├── slides.pptx
│   └── demo_video_link.txt
└── samples/
    └── README.md
```

## Suggested team contribution section
Replace these placeholders with actual names, roll numbers and real contributions:
- Member 1: application implementation and integration.
- Member 2: dataset research, testing and evaluation.
- Member 3: report and presentation.
- Member 4: documentation and demonstration.

Do not claim work that was not performed.
