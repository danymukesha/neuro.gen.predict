# NeuroGenPredict

A Genetic Risk Assessment Tool for Alzheimer's Disease.

## Installation
1. Clone or create the project directory.
2. Run `pip install -e .` to install the package.

## Usage
- Generate example data: `python examples/generate_example_data.py`
- Train and predict via CLI: `python examples/train_and_predict.py`
- Run the web app: `streamlit run app.py`

## Data Format
- Genotype CSV: Rows = samples, columns = variants (e.g., 'APOE_e4'), values = 0/1/2.
- Labels CSV: Column 'label' with 0 (control)/1 (AD case).
- Clinical CSV (optional): Columns like 'age', 'sex', 'education_years'.

For real data, obtain from public repositories like NIAGADS (requires approval). Simulated data is for demo only.

## For Publication/Patent/Startup
- The tool uses your ensemble method for innovation.
- Validate on real datasets you acquire.
- Extend as needed (e.g., add VCF parsing).

Author: Dany Mukesha
# neuro.gen.predict
