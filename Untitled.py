
# coding: utf-8

# # Sentiment Analysis and work cloud for positive and negitive words

# In[1]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag


# In[2]:


import nltk.classify.util
import nltk
nltk.download('movie_reviews')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')


# In[3]:


negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]


# In[4]:


negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)


# In[5]:


trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()


# In[6]:



test_data = "Bad"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
print(test_data_features)
print (classifier.classify(test_data_features))


# In[47]:


from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
wiki = TextBlob("I hate this library",analyzer=NaiveBayesAnalyzer())
print(wiki.sentiment)


# In[55]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer 
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores('Bad Management')
print(ss)


# In[2]:


lemmatizer = WordNetLemmatizer()


# In[3]:


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def clean_text(text):
    text = text.replace("<br />", " ")
    text = text.decode("utf-8")
    return text


# In[5]:


def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
    sentiment = 0.0
    tokens_count = 0
    text = clean_text(text)
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
    # negative sentiment
    return 0


# In[ ]:


# Since we're shuffling, you'll get diffrent results
print swn_polarity(train_X[0]), train_y[0] # 1 1
print swn_polarity(train_X[1]), train_y[1] # 0 0
print swn_polarity(train_X[2]), train_y[2] # 0 1
print swn_polarity(train_X[3]), train_y[3] # 1 1
print swn_polarity(train_X[4]), train_y[4] # 1 1`

