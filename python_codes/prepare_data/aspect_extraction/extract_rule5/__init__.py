from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_rule5(doc, ner_heads, prod_pronouns, sid, product_id):
    ## FIFTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    #Complement of a copular verb - A is a child of M with relationship of nsubj, while
    # M has a child with relationship of cop
    #Assumption - A verb will have only one NSUBJ and DOBJ
    rule5_pairs = []
    
    for token in doc:
        children = token.children
        A = "999999"
        buf_var = "999999"
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                if child.idx in ner_heads:
                    A = ner_heads[child.idx].text
                else:
                    A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "cop" and not child.is_stop):
                buf_var = child.text
                #check_spelling(child.text)

        if(A != "999999" and buf_var != "999999"):
            if A in prod_pronouns :
                A = product_id
            dict5 = {"noun" : A, 
                     "adj" : token.text, 
                     "rule" : 5, 
                     "polarity_nltk" : sid.polarity_scores(token.text)['compound'],
                     "polarity_textblob" : TextBlob(token.text).sentiment.polarity}
            rule5_pairs.append(dict5)
    return rule5_pairs