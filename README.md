# Defining Project Objectives:

The objective of this project was to develop a sentiment analysis and prediction system for reviews associated with Disneyland. This was achieved using the Python programming language and relevant machine learning libraries such as NLTK and Scikit-Learn. The development process involved collecting and processing the Disneyland Reviews dataset from Kaggle, which contains 42,657 entries and 6 columns: Review_ID, Rating, Year_Month, Reviewer_Location, Review_Text, and Branch.

# Data Preprocessing:

During this stage, text preprocessing techniques were applied using the Pandas library to manipulate and analyze the dataset. Only two columns, "Rating" and "Review_Text," were chosen as they were relevant to the project's goal. To ensure data integrity and analysis coherence, 23 duplicate entries were identified and removed. To label sentiments, a new column "Sentiment" was created, assigning the label -1 (negative) to reviews with ratings between 1 and 2, 1 (positive) to those with ratings between 4 and 5, and 0 (neutral) to others. A function for cleaning and tokenizing textual data was defined, removing punctuation, numbers, and stop words. Additionally, the TF-IDF technique from the Scikit-Learn library was applied to convert the review text into numerical representations. At the end of this stage, the dataframe consisted of 3 columns: "Rating," "Preprocessed Review_Text," and "Sentiment," with 42,633 entries.

# Implementation of Sentiment Analysis and Prediction Models:

Concerning the analysis models, a classification into multiple categories, i.e., positive, negative, neutral, was implemented. This process involved data processing and creating sentiment labels based on ratings. To predict sentiments based on preprocessed textual features, the dataset was split into 20% test data and 80% training data. Prediction models were implemented using machine learning algorithms such as logistic regression and support vector machine (SVM). For the logistic regression model, the solver was set to 'lbfgs,' regularization was of type L2 (penalty='l2'), and multi-class classification was specified using 'multinomial.' For the SVM model, the kernel parameter was set to 'linear.'

# Evaluation and Interpretation of Results:

For the logistic regression model (logistic_regression_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2')), the results were as follows:

Test set:

Accuracy: 0.8444939603612056
Precision: 0.8139817140232668
Recall: 0.8444939603612056
F1 Score: 0.8181743818434075

Training set:

Accuracy: 0.8905764381633731
Precision: 0.8821784353893216
Recall: 0.8905764381633731
F1 Score: 0.8757443034923221
The results indicate good performance on both datasets, with high accuracy, precision, recall, and F1 Score.

For the SVM model (svm_model = svm.SVC(kernel='linear', cache_size=2000)), the results were:

Test set:

Accuracy: 0.8449630585199953
Precision: 0.8135905657574153
Recall: 0.8449630585199953
F1 Score: 0.819275570899968

Training set:

Accuracy: 0.9106608807834399
Precision: 0.9068241664473374
Recall: 0.9106608807834399
F1 Score: 0.9002243717087699
The SVM model also demonstrated good overall performance on both datasets.

While both models show good performance, there are instances where the system does not correctly predict class membership. For example, a review expressing disappointment with a Disneyland visit was classified as positive by both models, despite human evaluation classifying it as negative.
