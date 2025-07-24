# NLP
Objective:
The goal of this project is to classify short text messages (tweets) into one of four classes:

1. Figurative
2. Irony
3. Regular
4. Sarcasm



Dataset:
The dataset train.csv contains 81,408 rows and 2 columns:

tweets: the actual tweet text

class: the target label

A sample of these tweets shows that they come with various forms of expression, such as sarcasm, irony, or figurative language.

To make the dataset manageable and balanced during experimentation, a random sample of 60,000 rows was dropped, leaving 21,408 rows (as per the image)

Preprocessing Pipeline:
Text Cleaning & Lemmatization:
Using NLTK, a function lemma(text) was created to clean and preprocess the tweet texts:

Removed URLs using regex.

Tokenized using TweetTokenizer.

Removed stopwords.

Performed POS tagging to get context-aware lemmatization.

Lemmatized using WordNetLemmatizer.


Train-Test Split:
From the cleaned dataset, input and target features were separated:

X: contains the "messages" column.
Y: contains the "class" column (encoded using LabelEncoder).

The data was then split into training and testing sets using an 80:20 split.

Feature Extraction:
To convert text into numerical form, TF-IDF vectorization was used:

TfidfVectorizer() was applied to the cleaned "messages" column for both training and testing data.

This captures the importance of each word in a message relative to the entire corpus.


Then I applied the ML Algorithms to predict the sentiment of the tweets. 