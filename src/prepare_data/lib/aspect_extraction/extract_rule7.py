from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_rule7(doc, ner_heads, prod_pronouns, sid, product_id):
    ## SEVENTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    ## ATTR - link between a verb like 'be/seem/appear' and its complement
    ## Example: 'this is garbage' -> (this, garbage)
    
    rule7_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                if child.idx in ner_heads:
                    A = ner_heads[child.idx].text
                else:
                    A = child.text
                # check_spelling(child.text)

            if((child.dep_ == "attr") and not child.is_stop):
                M = child.text
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns :
                A = product_id
            dict7 = {"noun" : A, 
                     "adj" : M, 
                     "rule" : 7, 
                     "polarity_nltk" : sid.polarity_scores(M)['compound'],
                     "polarity_textblob" : TextBlob(M).sentiment.polarity}
            rule7_pairs.append(dict7)
            
    return rule7_pairs