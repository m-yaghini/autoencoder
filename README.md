# Symmetric Autoencoder for Text Classification
Symmetric autoencoder using PyTorch with document classifier and data-scrapper for Wikipedia articles. 

## Results
Preliminary results are curated in the Jupiter notebook, `Results.ipynb`, using two sets of data (a pilot of 10 classes * 100 articles per class, and full data 10 classes * 1000 articles) and two preprocessing methods (word frequencies, word embeddings). A couple of baselines are also included for comparison. 

There is still a lot of room for hyper-parameter optimization. But the essentials are done. The model architecture is expandable by further work, through more complex architectures.

## Training
The size of the model is very manageable, but the reconstructed feature set is big. Data for most stages are included in `/data` directory: from raw scrapped text data (using the scrapper class in `/code/`) to processed data in numeric format with two mechanisms:  either word embeddings or frequencies.

You can run the code to train the model and do a basic classification. Configuration can be changed in the `run_model.py` script.

```
python run_model.py
```

Mohammad Yaghini, April 2018
