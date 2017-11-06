# USE PYTHON 3
import sys, os
from glob import glob
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import aspell
import chardet
from auxiliar import get_content

input_dir = sys.argv[1]

files = glob(os.path.join(input_dir, "*"))
sid = SentimentIntensityAnalyzer()
spell = aspell.Speller()

possible_sentiment = ['compound', 'neg', 'neu', 'pos']
possible_tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']

printheader = ["filename"] +\
                    [ s for s in possible_sentiment ] +\
                    [ s if s != '.' else 'PUNCT' for s in possible_tags ] +\
                    ["perc_not_aspell","avg_suggestion_length"] +\
                    ["entities_per_sentence","entities_per_word"] +\
                    ["avg_tree_height"]

print(",".join(printheader))

def count_entity_names(t):
    entity_names = 0

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names += 1
        else:
            for child in t:
                entity_names += count_entity_names(child)

    return entity_names

for f in files:
    content = get_content(f, htmlremover=None)

    lines_list = tokenize.sent_tokenize(content)
    nsentences = len(lines_list)
    ss_sum = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    known_pos = dict((t,0.0) for t in possible_tags)
    ntokens = 0
    novv = 0
    sum_suggestions = 0.0
    ne_detected = 0
    sum_height = 0

    for sent in lines_list:
        ss = sid.polarity_scores(sent)
        for sentiment in possible_sentiment:
            ss_sum[sentiment] += ss[sentiment]

        tags = nltk.pos_tag(tokenize.word_tokenize(sent))

        chunked_sentence = nltk.ne_chunk(tags, binary=True)
        ne_detected += count_entity_names(chunked_sentence)
        sum_height += chunked_sentence.height()

        for (w, t) in tags:
            known_pos[ nltk.tag.mapping.map_tag('en-ptb', 'universal', t) ] += 1
            try:
                correct = spell.check(w)
            except:
                continue #ignore this word

            if not correct:
                novv += 1
                sum_suggestions += len(spell.suggest(w))


        ntokens += len(tags)

    for sentiment in possible_sentiment:
        ss_sum[sentiment] = ss_sum[sentiment] / nsentences if nsentences > 0 else 0.0

    for pos in possible_tags:
        known_pos[pos] = known_pos[pos] / ntokens if ntokens > 0 else 0.0

    printvector = [os.path.basename(f)] +\
                        [ ("%.4f" % (ss_sum[s])) for s in possible_sentiment ] +\
                            [ ("%.4f" % (known_pos[s])) for s in possible_tags ] +\
                            [ "%.4f,%.4f" % (novv / ntokens if ntokens > 0 else 0.0, sum_suggestions / novv if novv > 0.0 else 0.0)] +\
                            [ "%.4f,%.4f" % (ne_detected / nsentences if nsentences > 0 else 0.0, ne_detected / ntokens if ntokens > 0 else 0.0) ] +\
                            [ "%.4f" % (sum_height / nsentences if nsentences > 0 else 0.0) ]

    print(",".join(printvector))



