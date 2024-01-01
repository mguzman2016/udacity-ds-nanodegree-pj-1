# udacity-ds-nanodegree-pj-1
This repository contains Python and Jupyter Notebook files for an in-depth analysis of the 2016 Seattle Airbnb dataset.

# Motivation
This project gives a detailed look into the Airbnb market in Seattle, using complete data analysis methods. This is part of a bigger learning project, showing how data science skills can be used in real life situations.

# Libraries used
This project utilizes several Python libraries for data processing, analysis, and visualization:

- numpy: For efficient numerical computations.
- pandas: For data manipulation and analysis.
- matplotlib: For creating static, interactive, and animated visualizations.
- seaborn: For high-level data visualization based on matplotlib.
- sklearn (Scikit-Learn): For machine learning and predictive data analysis.
- xgboost: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

# Files in the repository

## Section 1 - Project.ipynb
A Jupyter Notebook encompassing the complete data analysis process, including business understanding, data exploration, data preprocessing, exploratory data analysis, machine learning model building, and visualizations.

## helpers.py
Contains helper functions that streamline the data cleaning and transformation process.

## training_helpers.py
This file includes the ModelsTrainer class, which facilitates the training and evaluation of machine learning models. Key functionalities:
- Model training and testing with performance metrics (MSE, R^2).
- Feature importance plotting for model interpretation.
The class simplifies the model training process and evaluation, aiding in better understanding and optimization of the models.

# Summary of results
The analysis revealed:

- The most popular neighborhoods in Seattle based on Airbnb reviews.
- Seasonal variations in pricing, indicating the best times for guests to book and hosts to list.
- Key features influencing listing prices, such as the number of bedrooms and privacy level.

# Acknowledgements
This project was made possible thanks to the Airbnb dataset available on Kaggle, which includes detailed listings, reviews, and pricing information for Seattle accommodations in 2016. The project is part of an educational pursuit, showcasing the application of data science techniques in real-world scenarios