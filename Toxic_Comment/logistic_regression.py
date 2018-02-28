import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.utils import shuffle
from xgboost import XGBClassifier

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_names_lf = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
#train_toxic = train[train[class_names_lf].sum(axis=1) > 0]  # lower freq offensive entries

#print("Upsampling...")
#comments_toxic = list()
#for comment in train_toxic["comment_text"]:
#    tokens = comment.split()
#    half_tokens = int(len(tokens)/2)
#    comments_toxic.append(" ".join(tokens[0:half_tokens]))

#train_toxic["commment_text"] = comments_toxic
#train = pd.concat([train, train_toxic])
print(len(train))

#print("Shuffle...")
#train = shuffle(train)

train_text = train['comment_text']
test_text = test['comment_text']

vectorizer_text = pd.concat([train_text, test_text])
    
print("Vectorize...")
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
#    stop_words='english',
    ngram_range=(1, 3),
    max_features=25000)
word_vectorizer.fit(vectorizer_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print("Vectorize...")
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
#    stop_words='english',
    analyzer='char',
    ngram_range=(2, 6),
    max_features=35000)
char_vectorizer.fit(vectorizer_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features], format='csr')
test_features = hstack([test_char_features, test_word_features], format='csr')

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
print("Start training...")
for class_name in class_names:
    print("Train class: ", class_name) 
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)