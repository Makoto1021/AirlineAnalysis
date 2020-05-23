#import enchant
import dataiku
import ast
from aspect_extraction.extract_rule1 import extract_rule1
from aspect_extraction.extract_rule2 import extract_rule2
from aspect_extraction.extract_rule3 import extract_rule3
from aspect_extraction.extract_rule4 import extract_rule4
from aspect_extraction.extract_rule5 import extract_rule5
from aspect_extraction.extract_rule6 import extract_rule6
from aspect_extraction.extract_rule7 import extract_rule7

prod_pronouns = dataiku.get_custom_variables(typed=True)['prod_pronouns']

def apply_extraction(row, nlp, sid, text_column, review_id, product_id):
    review_body = row[text_column]
    review_id = row[review_id]
    # review_marketplace = row['marketplace']
    # customer_id = row['customer_id']
    product_id = row[product_id]
    # product_parent = row['product_parent']
    # product_title = row['product_title']
    # product_category = row['product_category']
    # date = str(row['review_date'])
    # star_rating = row['star_rating']
    # url = add_amazonlink(product_id)

    doc=nlp(review_body)
    ner_heads = {ent.root.idx: ent for ent in doc.ents}

    rule1_pairs = extract_rule1(doc, ner_heads, prod_pronouns, sid, product_id)
    rule2_pairs = extract_rule2(doc, ner_heads, prod_pronouns, sid, product_id)
    rule3_pairs = extract_rule3(doc, ner_heads, prod_pronouns, sid, product_id)
    rule4_pairs = extract_rule4(doc, ner_heads, prod_pronouns, sid, product_id)
    rule5_pairs = extract_rule5(doc, ner_heads, prod_pronouns, sid, product_id)
    rule6_pairs = extract_rule6(doc, ner_heads, prod_pronouns, sid, product_id)
    rule7_pairs = extract_rule7(doc, ner_heads, prod_pronouns, sid, product_id)

    aspects = []
    aspects = rule1_pairs + rule2_pairs + rule3_pairs +rule4_pairs +rule5_pairs + rule6_pairs + rule7_pairs
    
    dic = {"review_id" : review_id , 
           "aspect_pairs" : aspects, 
           # "review_marketplace" : review_marketplace,
           # "customer_id" : customer_id, 
           "product_id" : product_id, 
           # "product_parent" : product_parent,
           # "product_title" : product_title, 
           # "product_category" : product_category, 
           # "date" : date, 
           # "star_rating" : star_rating, 
           # "url" : url
          }

    return dic
