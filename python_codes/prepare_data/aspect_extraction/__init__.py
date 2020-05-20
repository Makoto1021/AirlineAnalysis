from aspect_extraction.aspect_extraction import aspect_extraction

if __name__ == '__main__' :
    # nlp = init_spacy()
    a = aspect_extraction(nlp, sid)

    # USE THIS IF YOU WANT TO SEE THE ASPECTS IN A FILE
    # with open('your_file.txt', 'w') as f:
    #     for item in a:
    #         f.write("%s\n" % item)
