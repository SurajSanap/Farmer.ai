# CultiVision

## Problem Definition
* Farmers face challenges in choosing the right crop based on soil characteristics, leading to reduced productivity and soil degradation.
* Lack of awareness about soil requirements, organic fertilizers, and standard fertilizers contributes to imbalanced fertilization and nutrient management issues.
* Annual crop losses of Rs. 50,000 crore occur due to pests in India.
  
![Screen Recording 2024-11-23 193838](https://github.com/user-attachments/assets/328fff0d-4ad5-4dd3-b8c5-c1a44fe2fae4)



## Objective
* Implement precision agriculture to guide farmers in crop selection and increase productivity.
* Propose an ensemble model using SVM, Random Forest, Naive Bayes, and kNN for accurate crop recommendation.
* Recommend fertilizers based on N, P, K values and crop types.
* Identify pests using a DL model and recommend pesticides adhering to ISO standards.

## Methodology
### Crop Recommendation
1. Acquire Dataset
2. Input Values
3. Train ML Model & Create .pkl file
    1. Ensemble Technique with Majority Voting
        1. SVM
        2. Random Forest
        3. Naive Bayes
        4. kNN
4. Crop Recommendation
    1. Load .pkl model file for crop recommendation.

### Fertilizer Recommendation
1. Acquire Dataset
    1. NPK values for different crops
    2. Sources: The Fertilizer Association of India, Indian Institute of Water Management, etc.
2. Input Values
3. Determine Difference between Desired & Actual
    1. High
    2. Low
    3. Optimal
4. Dictionary-based Suggestions from Verified Sources.

## Pesticide Recommendation
1. Acquire Data
    1. Image Scraping from Google Images
    2. Provide Pest Labels
2. Data Cleaning & Augmentation
    1. Remove Unnecessary Content
    2. Augment Dataset for Variability
3. DL Model Creation
4. Pest Detection & Pesticide Recommendation

## Conclusion
The solution aims to enhance farmer productivity, reduce soil degradation, and provide informed advice on fertilizers and crop selection. It addresses the critical issue of pest control, benefitting both farmers and the environment.

## How to Run Locally?

1. Download PyCharm.
2. Create a new environment using the command: `conda create -n env_name python==3.7.0`.
3. Activate the environment using the command: `conda activate env_name`.
4. Install requirements by navigating to the project folder and typing: `pip install -r requirements.txt`.
5. Run `app.py` using the command: `python app.py`.
