#load string cleaning modules
import re, string
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def clean_grammer(value:list,
                  stemmer_=SnowballStemmer('english',ignore_stopwords=False),
                  lemmatizer_=WordNetLemmatizer()):
  clean_value=list()
  for word,pos in pos_tag(value):
    if stemmer_:word=stemmer_.stem(word)
    if pos.startswith('J'): word=lemmatizer_.lemmatize(word,'r')
    elif pos.startswith('R'): word=lemmatizer_.lemmatize(word,'s')
    elif pos.startswith('V'): word=lemmatizer_.lemmatize(word,'v')
    elif pos.startswith('N'): word=lemmatizer_.lemmatize(word,'n')
    else: word=lemmatizer_.lemmatize(word,'n')
    clean_value.append(word)
  return clean_value

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean(value:str,tags2remove:list=['http[\S]+','@[\S]+','&amp[\S]+'],
          tag2convert:dict={'positive ':[':)',';',':-)',':p',':D'],
                            'negative ':[':(',';(',':-(']},
          cleaner:str='[^a-zA-Z ]',
          stopwords:list=stopwords.words(fileids='english'),
          minimum_word_length:int=3,call_clean_grammer:bool=True,
          clean_grammer_method=clean_grammer):
  if isinstance(value,(list,tuple)):
    value=[clean(sentc) for sentc in value]
    return value
  for tag in tags2remove:
    value= re.sub(tag,'',value)
  for tag_name,tags in tag2convert.items():
    for tag in tags:
      value=value.replace(tag,tag_name)
  value=re.sub(cleaner,'',value)
  value=value.casefold()
  value=word_tokenize(value)
  value=[word for word in value if word not in stopwords]
  value=[word for word in value if not len(word)<minimum_word_length]
  if call_clean_grammer: return clean_grammer_method(value)
  else: return value

import pickle
with open('nltk.nb.model',mode='rb')as model_file:
      model=pickle.load(model_file)

tweet=input('Enter Tweet to check: ')

prediction=model.classify(dict([(word,True)for word in clean(tweet)]))

print(f'The Given tweet -> \n\t"{tweet}"\n\t is "{prediction}".')
