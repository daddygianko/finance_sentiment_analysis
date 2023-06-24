from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
import os
cwd = os.getcwd()
os.environ["MODEL_DIR"] = cwd+'//descargas//.'
import nlpaug
import nlpaug.augmenter.word as naw
from nlpaug.util import Action
#from googletrans import Translator

stop_words = []
for w in stopwords.words('english'):
    stop_words.append(w)


##### TECNICAS EDA #####

# 1. REEMPLAZAR CON SINONIMOS
def reemplazar_sinonimos(lista_texto, max_cambios=1, n_cambios_random=False, repeticiones=1):
    """_summary_

    Args:
        lista_texto (_type_): _description_
        max_cambios (int, optional): _description_. Defaults to 1.
        n_cambios_random (bool, optional): _description_. Defaults to False.
        repeticiones (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if(n_cambios_random):
        lista = [str(__synonym_replacement(sent, random.randint(1, max_cambios))) for sent in lista_texto for _ in range(repeticiones)]
    else:   
        lista = [str(__synonym_replacement(sent, max_cambios)) for sent in lista_texto for _ in range(repeticiones)]
    return lista

def __get_synonyms(word):
    
    synonyms = set()
    
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def __synonym_replacement(words, n):
    """_summary_

    Args:
        words (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = __get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

#2. BORRAR PALABRA AL AZAR
def borrar_random(lista_texto, max_prob=0.1, prob_borrar_random=False, repeticiones=1):
    """_summary_

    Args:
        lista_texto (_type_): _description_
        max_prop (float, optional): _description_. Defaults to 0.1.
        prop_borrar_random (bool, optional): _description_. Defaults to False.
        repeticiones (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if(prob_borrar_random):
        lista = [str(__random_deletion(sent, round(random.uniform(0.05, max_prob), 2))) for sent in lista_texto for _ in range(repeticiones)]
    else:   
        lista = [str(__random_deletion(sent, max_prob)) for sent in lista_texto for _ in range(repeticiones)]
    
    return lista

def __random_deletion(words, p):
    """_summary_

    Args:
        words (_type_): _description_
        p (_type_): _description_

    Returns:
        _type_: _description_
    """
    words = words.split()
    
    #obviously, if there's only one word, don't delete it
    if len(words) <= 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    
    return sentence

#3. Cambiar el orden de las palabras al azar
def orden_random(lista_texto, max_cambios=1, n_cambios_random=False, repeticiones=1):
    """_summary_

    Args:
        lista_texto (_type_): _description_
        max_cambios (int, optional): _description_. Defaults to 1.
        n_cambios_random (bool, optional): _description_. Defaults to False.
        repeticiones (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if(n_cambios_random):
        lista = [str(__random_swap(sent, random.randint(1, max_cambios))) for sent in lista_texto for _ in range(repeticiones)]
    else:   
        lista = [str(__random_swap(sent, max_cambios)) for sent in lista_texto for _ in range(repeticiones)]
    return lista

def __swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def __random_swap(words, n):
    """_summary_

    Args:
        words (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    words = words.split()
    new_words = words.copy()
    # n is the number of words to be swapped
    for _ in range(n):
        new_words = __swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence

#4. Insertar sinonimos en posiciones al azar
def insercion_random(lista_texto, max_cambios=1, n_cambios_random=False, repeticiones=1):
    """_summary_

    Args:
        lista_texto (_type_): _description_
        max_cambios (int, optional): _description_. Defaults to 1.
        n_cambios_random (bool, optional): _description_. Defaults to False.
        repeticiones (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if(n_cambios_random):
        lista = [str(__random_insertion(sent, random.randint(1, max_cambios))) for sent in lista_texto for _ in range(repeticiones)]
    else:   
        lista = [str(__random_insertion(sent, max_cambios)) for sent in lista_texto for _ in range(repeticiones)]
    return lista

def __random_insertion(words, n):
    """_summary_

    Args:
        words (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        __add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def __add_word(new_words):
    
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = __get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
        
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


    ##### NLPAUG #####

#1. Reemplazar letras al azar en las palabras para simular errores de escritura
def aumentar_mispelling(lista_texto, repeticiones=1):
    lista = [__spelling_aug(texto, repeticiones) for texto in lista_texto]
    lista = [sent for elem in lista for sent in elem]
    return lista
    
def __spelling_aug(text, n=3):
    aug = naw.SpellingAug()
    augmented_texts = aug.augment(text, n)
    return augmented_texts

#2. Insertar palabras usando RoBERTa's contextual word embeddings
def insertar_word_emb(lista_texto, repeticiones=1):
  lista = [__insert_word_embs_aug(texto, repeticiones) for texto in lista_texto]
  lista = [sent for elem in lista for sent in elem]
  return lista

def insertar_word_emb2(lista_texto, repeticiones=1):
  lista=[]
  for sent in lista_texto:
      for _ in range(repeticiones):
          lista.append(__insert_word_embs_aug(sent, 1))
  return lista

def __insert_word_embs_aug(text, n=3):
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="insert")
    augmented_text = aug.augment(text, n)
    return augmented_text

#3. Reemplazar palabras usando RoBERTa's contextual word embeddings
def reemplazar_word_emb(lista_texto, repeticiones=1):
  lista = [__subs_word_embs_aug(texto, repeticiones) for texto in lista_texto]
  lista = [sent for elem in lista for sent in elem]
  return lista

def reemplazar_word_emb2(lista_texto, repeticiones=1):
  lista=[]
  for sent in lista_texto:
      for _ in range(repeticiones):
          lista.append(__subs_word_embs_aug(sent, 1))
  return lista

def __subs_word_embs_aug(text, n=3):
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
    augmented_text = aug.augment(text, n)
    return augmented_text

#4. Traducir a un idioma distinto yluego volverlo a traducir al inglés
"""
langs = ['es', 'de', 'it', 'fr', 'ja', 'ru']
translator=Translator()

def aumentar_backtranslate(lista_texto, repeticiones=2):
    list_lang = random.sample(langs, min(len(langs), repeticiones))
    lista = [__translate_aug(sent, lang) for sent in lista_texto for lang in list_lang]
    lista = list(filter(lambda x: len(x)>1, lista))
    return lista

def __translate_aug(text, lang):
    try:
      translation = translator.translate(text, src='en', dest=lang).text
      back_translation = translator.translate(translation, src=lang, dest='en').text
    except:
      return ''
    else:
      return back_translation
"""

def aumentar_backtranslate(lista_texto):
    lista1=[__back_translation_aug_de(sent) for sent in lista_texto]
    lista2=[__back_translation_aug_ru(sent) for sent in lista_texto]
    lista = lista1+lista2
    lista = list(filter(lambda x: len(x)>1, lista))
    return lista

def __back_translation_aug_de(text):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )
    try:
        augmented_text = back_translation_aug.augment(text)
    except:
        return ''
    else:
        return augmented_text

def __back_translation_aug_ru(text):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-ru', 
        to_model_name='facebook/wmt19-ru-en'
    )
    try:
        augmented_text = back_translation_aug.augment(text)
    except:
        return ''
    else:
        return augmented_text
