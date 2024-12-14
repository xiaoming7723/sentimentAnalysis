
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import emoji
from nltk.corpus import stopwords
from num2words import num2words
from nltk import pos_tag
from nltk.corpus import wordnet
import string

# import vectorizer
with open('tfidf_stem_sen.pkl', 'rb') as file:
    tfidf_sen = pickle.load(file)
        
# import model
with open('Logistic Regression - S_Stem.pkl', 'rb') as file:
    model_sen = pickle.load(file)
    
with open('Bernoulli Naive Bayes - Augment_Lemmatize_sar.pkl', 'rb') as file:
    model_sar = pickle.load(file)

with open('tfidf_augment_sar.pkl', 'rb') as file:
    tfidf_sar = pickle.load(file)

    
keep_stopwords = {'not', 'no', "don't", "didn't", "doesn't", "hadn't", "haven't", "isn't",
                  "mustn't", "shouldn't", "wasn't", "weren't", "won't", "i", "you're", "you've",
                  "you're", "you've", "you'll", "you'd", "she's", "that'll", 'until', 'food',
                  'place', 'service', 'good', 'nice', 'delicious', 'restaurant', 'eatery',
                  'dining', 'experience', 'meal', 'menu', 'dish', 'cuisine', 'server', 'waiter',
                  'waitress', 'staff', 'atmosphere', 'ambiance', 'decor', 'taste', 'flavor',
                  'presentation', 'price', 'value', 'pricey', 'affordable', 'cost', 'money',
                  'clean', 'hygiene', 'hygienic', 'spacious', 'cozy', 'crowded', 'busy', 'empty',
                  'reservation', 'wait', 'waiting', 'quick', 'fast', 'slow', 'speed', 'fresh',
                  'overcooked', 'undercooked', 'bland', 'spicy', 'salty', 'sweet', 'sour',
                  'crispy', 'tender', 'juicy', 'portion', 'size', 'huge', 'small', 'large',
                  'recommend', 'suggest', 'try', 'must', 'definitely', 'absolutely', 'probably',
                  'likely', 'unlikely', 'maybe', 'perhaps', 'probably', 'possibly', 'certainly',
                  'surely', 'love', 'like', 'dislike', 'hate', 'enjoy', 'prefer', 'favor',
                  'impressed', 'satisfied', 'happy', 'unhappy', 'disappointed', 'pleasant',
                  'unpleasant', 'friendly', 'rude', 'polite', 'attentive', 'negligent', 'helpful',
                  'accommodating', 'knowledgeable', 'courteous', 'recommendation', 'favorite','good','ok','nice','fast'}

stopwords_list = set(stopwords.words('english'))
stopwords_list -= keep_stopwords

abbreviations = {'fyi': 'for your information',
                 'lol': 'laugh out loud',
                 'loza': 'laughs out loud',
                 'lmao': 'laughing',
                 'rofl': 'rolling on the floor laughing',
                 'vbg': 'very big grin',
                 'xoxo': 'hugs and kisses',
                 'xo': 'hugs and kisses',
                 'brb': 'be right back',
                 'tyt': 'take your time',
                 'thx': 'thanks',
                 'abt': 'about',
                 'bf': 'best friend',
                 'diy': 'do it yourself',
                 'faq': 'frequently asked questions',
                 'fb': 'facebook',
                 'idk': 'i don\'t know',
                 'asap': 'as soon as possible',
                 'syl': 'see you later',
                 'nvm': 'never mind',
                 'frfr':'for real for real',
                 'istg':'i swear to god',
    }

# remove punctuations, numbers, and stopwords
def clean_text_sar(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

# remove punctuations, numbers, and stopwords
def clean_text(sentences):
    # convert text to lowercase
    text = sentences.lower()

    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    
    # remove mentions
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))

    # replace emoji with text
    text = emoji.demojize(text)
    text = re.sub(r'[:_]', ' ', text)
    text = ' '.join(text.split())

    # replace time stamp
    time_reg = re.compile("(\d{1,2}:)?\d{1,2}:\d{2}(\.\d{1,3})?")
    text = time_reg.sub(r'',text)

    # contractions
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    
    # abbreviation
    text = replace_abbreviations(text)
    
    # Replace ‘’ with ''
    text = re.sub(r'[‘’]','\'', text)

    # Replace “” with ""
    text = text.replace('“', '"').replace('”', '"')

    # remove text in square brackets
    text = re.sub('\[.*?\]', '', text)

    # removing punctuations and replace with space
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    # Substitute ordinal numbers like "1st", "2nd", "3rd" with words
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', lambda match: ordinal_to_word(match.group(1)), text)

    # removing words containing digits
    text = re.sub(r'\d+', '', text)
    
    # remove duplicated charcter to only one
    text = re.sub(r'([a-zA-Z])\1{2,}','\1', text) 

    # Join the words
    text = ' '.join([word for word in text.split()
                     if word not in stopwords_list])
    return text

def ordinal_to_word(number_str):
    # Convert the string representation of the number to an integer
    number = int(number_str)

    return(num2words(number, to ="ordinal"))

def replace_abbreviations(text):
    for abbr, full_form in abbreviations.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
    return text


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

porter_stemmer = PorterStemmer()
def stem_sentence(text):
    new_sen = []
    if isinstance(text, str):
        words = word_tokenize(text)
        pos_tagged = pos_tag(words)
        wordnet_tagged = list(map(lambda x: (x[0],pos_tagger(x[1])), pos_tagged)) # POS Tagging & Reducing
        for word, tag in wordnet_tagged:
            if tag is None:
                new_sen.append(word)
            else:
                new_sen.append(porter_stemmer.stem(word,tag))                   # stemming
    
    return ' '.join(new_sen)

def predict_sarcasm(text):
    cleaned_text = clean_text_sar(text)
    text_vectorized = tfidf_sar.transform([cleaned_text])
    sarcasm_prediction = model_sar.predict(text_vectorized)
    return sarcasm_prediction[0]

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    stemmed_text = stem_sentence(cleaned_text)
    text_vectorized = tfidf_sen.transform([stemmed_text])
    sentiment_prediction = model_sen.predict(text_vectorized)
    
    return sentiment_prediction[0] 


enter_trigger = """
<script>
document.addEventListener('keydown', function(event) {
    if (event.code === 'Enter') {
        document.querySelector('.streamlit-button').click();
    }
});
</script>
"""

st.markdown(enter_trigger, unsafe_allow_html=True)

st.title('Sentiment Analysis with Sarcasm Detection')

# User input
user_input = st.text_area('Enter your text here:')

# Make prediction
if st.button('Predict'):
    if user_input:
        text_for_sarcasm = user_input
        predicted_sentiment = predict_sentiment(user_input)
        
        if(predicted_sentiment == 1.0):

            if predict_sarcasm(user_input) == 1.0:
                sarcasm = True
            elif predict_sarcasm(user_input) == 0.0:
                sarcasm = False
            
            if sarcasm:
                st.write('Negative')
            else:
                st.write('Positive')
            
        else:
            st.write('Negative') 

    else:
        st.warning('Please enter some text before predicting.')
