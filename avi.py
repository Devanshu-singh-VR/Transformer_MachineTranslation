import spacy

eng = spacy.load("en_core_web_sm")
ger = spacy.load("de_core_news_sm")

def tokenizer_eng(text):
    return [tok.text for tok in eng.tokenizer(text)]

def tokenizer_ger(text):
    return [tok.text for tok in ger.tokenizer(text)]

print(tokenizer_eng('hello how are you'))

