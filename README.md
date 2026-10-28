# Heart Disease Risk Prototype

A minimal heart disease risk prototype with three selectable datasets and a separate model for each.

> Academic prototype. Not a medical diagnostic tool.

## Files

- `train.py`: trains a Random Forest for each dataset, prints metrics, saves `models.joblib`
- `dataset_config.py`: dataset feature, target, and display configuration shared by training and the app
- `app.py`: Streamlit UI for selecting a dataset and entering its corresponding features
- `data/heart.csv`: 918-row heart disease dataset
- `data/Cardiovascular_Disease_Dataset.csv`: 1,000-row cardiovascular disease dataset
- `data/CVD Dataset.csv`: 1,529-row CVD risk-level dataset

## Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The app's dataset selector switches between the three dataset-specific models and
shows the corresponding inputs and prediction target.
