# Cascade Method: Occupational Classification Semantic Matcher

## Overview

This Python tool harmonizes occupational classification systems (e.g., SOC 2020, ISCO-08, NHS classifications) using Natural Language Processing (NLP) models. It implements a cascade methodology combining Bag of Words (BoW), TF-IDF, and SBERT semantic embeddings to find semantic matches between job titles across different classification systems.

## Features

- **Multiple NLP Methods**: BoW, TF-IDF, SBERT semantic embeddings, and ensemble combinations
- **Advanced Preprocessing**: Tokenization, stopwords removal, and lemmatization using NLTK
- **Threshold Sensitivity Analysis**: Evaluates performance at 30%, 40%, and 50% thresholds
- **Ensemble Ablation Study**: Tests 8 different weight configurations
- **Manual Validation Support**: Creates stratified samples for inter-annotator agreement analysis
- **Comprehensive Reporting**: Generates detailed reports with precision, recall, F1 scores

---

## Requirements

### Python Version
- Python 3.8 or higher (tested with Python 3.11.4)

### Required Libraries

```bash
pip install pandas numpy scikit-learn sentence-transformers nltk openpyxl
```

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥2.0.0 | Numerical operations |
| scikit-learn | ≥1.0.0 | BoW, TF-IDF, cosine similarity, metrics |
| sentence-transformers | ≥5.0.0 | SBERT semantic embeddings |
| nltk | ≥3.8.0 | Tokenization, stopwords, lemmatization |
| openpyxl | ≥3.0.0 | Excel file reading |

---

## Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv cascade_env

# Activate (Windows)
cascade_env\Scripts\activate

# Activate (macOS/Linux)
source cascade_env/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn sentence-transformers nltk openpyxl
```

### Step 3: Download NLTK Resources

The script automatically downloads required NLTK resources on first run, but you can pre-download them:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Data Preparation

### Input File Format

Prepare an Excel file (`.xlsx`) with occupation data. Each classification system should be in a separate sheet.

**Example structure:**

| Sheet Name | Required Columns |
|------------|------------------|
| ISCO-08 | `unit` (ID), `description` (job title text) |
| SOC-2020 | `Unit_Group` (ID), `Group_Title` (job title text) |

### Column Requirements

- **ID Column**: Unique identifier for each occupation (numeric or alphanumeric)
- **Text Column**: Job title or description text for semantic matching

---

## Configuration

### Step 1: Update File Paths

Edit the `load_and_process_tables()` function (around line 1347) to set your file path:

```python
file_path = 'C:/your/path/to/data.xlsx'  # Change this to your file path
```

### Step 2: Configure Column Names

Update column names to match your data (around line 1366):

```python
TEXT_COLUMN_TABLE1 = 'description'      # Text column in Sheet 1
TEXT_COLUMN_TABLE2 = 'Group_Title'      # Text column in Sheet 2
ID_COLUMN_TABLE1 = 'unit'               # ID column in Sheet 1
ID_COLUMN_TABLE2 = 'Unit_Group'         # ID column in Sheet 2
```

### Step 3: Configure Sheet Names

Update sheet names in the data loading section (around line 1350):

```python
table1 = pd.read_excel(file_path, sheet_name='ISCO-08')   # First sheet
table2 = pd.read_excel(file_path, sheet_name='SOC-2020')  # Second sheet
```

### Step 4: Configure Output Directory

Update the output path for results (around line 1636):

```python
output_base = 'C:/your/output/directory/'  # Change this
```

---

## Running the Analysis

### Basic Execution

```bash
python cascade_method.py
```

### What Happens on First Run

1. **NLTK Resources**: Downloads tokenizers, stopwords, and WordNet (automatic)
2. **SBERT Model**: Downloads `paraphrase-multilingual-MiniLM-L12-v2` (~500MB)
3. **Model Cache**: Saves model locally in `./local_models/` for future runs

**Note**: First run takes longer (5-10 minutes) due to model download. Subsequent runs are faster.

---

## Analysis Pipeline

The script executes a 9-step analysis pipeline:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load Excel data | Confirmation of loaded records |
| 2 | Configure parameters | Column and ID mappings |
| 3 | Initialize matcher | Model configuration report |
| 4 | Threshold sensitivity | Results at 30%, 40%, 50% thresholds |
| 4.5 | Method comparison | BoW, TF-IDF, Semantic results |
| 5 | Ensemble ablation | 8 weight configurations tested |
| 6 | Generate reports | JSON reports in `./reports/` |
| 7 | Runtime summary | Timing statistics |
| 8 | Save results | CSV files for each analysis |
| 9 | Validation sample | 100-record stratified sample |

---

## Output Files

### Results Directory

```
output_directory/
├── method_bow_results.csv           # Bag of Words matches
├── method_tfidf_results.csv         # TF-IDF matches
├── method_semantic_results.csv      # SBERT semantic matches
├── threshold_30%_results.csv        # 30% threshold results
├── threshold_40%_results.csv        # 40% threshold results
├── threshold_50%_results.csv        # 50% threshold results
├── ablation_*.csv                   # Ensemble weight configurations
└── validation_sample_for_annotation.csv  # Manual validation sample
```

### Reports Directory

```
./reports/
├── comprehensive_report_YYYYMMDD_HHMMSS.json
└── validation_report_YYYYMMDD_HHMMSS.txt
```

### Cache Directories

```
./local_models/          # Downloaded SBERT model (reused)
./embeddings_cache/      # Cached embeddings (speeds up reruns)
```

---

## Understanding the Output

### Results CSV Columns

| Column | Description |
|--------|-------------|
| `id_table1` | ID from first classification |
| `occupation_table1` | Job title from first classification |
| `id_table2` | ID from second classification |
| `occupation_table2` | Job title from second classification |
| `similarity_percentage` | Match score (0-100%) |
| `match_ranking` | Rank among top matches |
| `method_used` | NLP method (bow/tfidf/semantic/all) |
| `similarity_bow` | BoW similarity score |
| `similarity_tfidf` | TF-IDF similarity score |
| `similarity_semantic` | SBERT similarity score |

### Confidence Bands

| Band | Similarity Range | Interpretation |
|------|------------------|----------------|
| High | ≥80% | Automatic acceptance recommended |
| Medium | 60-79% | Expert review recommended |
| Low | 30-59% | Manual classification required |
| Reject | <30% | No reliable match |

---

## Manual Validation Process

### Step 1: Annotate the Validation Sample

Open `validation_sample_for_annotation.csv` and fill in:

| Column | Values | Description |
|--------|--------|-------------|
| `annotator_1` | 1 or 0 | First annotator: 1=correct, 0=incorrect |
| `annotator_2` | 1 or 0 | Second annotator: 1=correct, 0=incorrect |
| `gold_standard` | 1 or 0 | Final agreed label after discussion |
| `notes` | text | Optional comments |

### Step 2: Calculate Validation Metrics

After annotation, run:

```python
# Load annotated file
annotations = pd.read_csv('validation_sample_for_annotation.csv', encoding='utf-8-sig', sep=';')

# Calculate inter-annotator agreement
agreement = calculate_inter_annotator_agreement(annotations)

# Calculate validation metrics
metrics = calculate_validation_metrics(annotations, 'Semantic Embeddings')

# Generate report
generate_validation_report(agreement, metrics)
```

### Metrics Reported

- **Agreement Rate**: Percentage of matching annotations
- **Cohen's Kappa**: Inter-annotator reliability (0-1)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

---

## Customization Options

### Change SBERT Model

```python
matcher = OccupationSemanticMatcher(
    model_name='all-MiniLM-L6-v2',  # Faster, English-only
    # model_name='paraphrase-multilingual-MiniLM-L12-v2',  # Default, multilingual
)
```

### Adjust Preprocessing

```python
matcher = OccupationSemanticMatcher(
    use_advanced_preprocessing=True,   # Enable/disable NLTK preprocessing
    language='english',                 # 'spanish', 'french', 'german', etc.
)
```

### Modify Ensemble Weights

```python
results = matcher.find_best_matches(
    ...,
    method='all',
    ensemble_weights=(0.25, 0.25, 0.50)  # (BoW, TF-IDF, Semantic)
)
```

### Change Threshold

```python
results = matcher.find_best_matches(
    ...,
    threshold=0.4,  # 40% minimum similarity
    top_n=3,        # Return top 3 matches per occupation
)
```

---

## Troubleshooting

### SSL Certificate Errors

The script includes automatic SSL configuration for restricted environments. If issues persist:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Memory Issues

For large datasets, reduce batch size:

```python
matcher = OccupationSemanticMatcher(
    batch_size=16,  # Reduce from default 32
)
```

### Model Download Failures

If model download fails, manually download and place in `./local_models/`:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save('./local_models/paraphrase-multilingual-MiniLM-L12-v2')
```

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Pooling | Mean pooling |
| Max Tokens | 128 |
| Batch Size | 32 |
| Random Seed | 42 |
| Embedding Dimension | 384 |

---

## Citation

If you use this code in your research, please cite:

```
Cascade Method for Occupational Classification Harmonization
University of Stirling - M.Sc. Data Science for Business
```

---

## License

This code is provided for academic and research purposes.

---

## Contact

For questions or issues, please refer to the GitHub repository or contact the authors.
