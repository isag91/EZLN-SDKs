import pandas as pd
import spacy
import re
nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')
nlp.max_length = 3500000

#########################
#Identify potential SDKs#
#########################

#keywords_SE.csv contains the statistical keywords extracted by Sketch Engine when comparing the whole corpus to a general Spanish language corpus
kwrd=pd.read_csv('keywords_SE.csv', skiprows=[0, 1])
keywords=kwrd['Item'].to_list()

#filtering keywords who are not nouns
#wrd=wrd[wrd['Score']>=1.00]
keywords=[w.text for wl in keywords for w in nlp(wl) if w.pos_ == "NOUN"]


#filtering keywords which are stopwords or do not exist
del_list=['ese', 'ahí', 'nos', 'ni', 'qué', 'l', 're', 'nam', 'c', 'do', 'cia', 'po', 'ó', 'ra', 'to', 'r', 'co', 'no', 'n', 'ca', 'mo', 'ta', 'www', 'nes', 'org', 'acá', 'ah', 'tra', 'so', 'd', 'y', 'pa', 'oea', 'lo', 'le', 'ro', 'ff', 'na', 's', 'pro', 'pue', 'tlc', 'tes', 'mos', 'amo', 'pag', 'pcp', 'ma', 'tos', 'aa', 'blo', 'que', 'tal', 'tar', 't', 'pre', 'im', 'gbi', 'per', 'tro', 'vil', 'lu', 'go', 'el', 'cha', 'ya', 'tas', 'gue', 'nal', 'ne', 'lar', 'así', 'cen', 'zas', 'bre', 'ara', 'ba', 'dis', 'cut', 'em', 'dih', 'or', 'fal', 'dih', 'ria', 'mm', 'vo', 'rra', 'a', 'cos', 'gn', 'se', 'nor', 'pe', 'más', 'za', 'bur', 'uni', 'ses', 'ga', 'gua', 'sdr', 'faz', 'ral', 'cla', 'dor', 'v', 'nas', 'ob', 'tam', 'ras', 'cam', 'up', 'mmh', 'an', 'nue', 'rá', 'í', 'ac', 'ble', 'cer', 'gro', 'rio', 'ri', 'htm', 'sas', 'tre', 'cho', 'i', 'pos', 'fes', 'fun', 'ins', 'm-l', 'bm', 'feu', 'lle', 'gar', 'li', 'nar', 'mu', 'ama', 'dea', 'sm', 'reo', 'zar', 'men', 'tie', 'fac', 'sec', 'in', 'ife', 'pu', 'ón', 'cio', 'moa', 'ini', 'am', 'ol', 'fa', 'afi', 'rue', 'fao', 'ar', 'cc', 'z', 'hh', 'tor', 'eu', 'num', 'cu', 'res', 'ter', 'dd', 'ban', 'bid', 'au', 'we', 'ci', 'of', 'fi', 'sa', 'be', 'jo', 'dió', 'car', 'ría', 'pm', 'der', 'il', 'dic', 'iu', 'ap', 'ei', 'xxi', 'bla', 'ene', 'ia', 'ce', 'via', 'hrs', 'cal', 'e', 'fué', 'can', 'ad', 'fax', 'por', 'ano', 'iv', 'pib', 'sub', 'du', 'g', 'uu', 'sos', 'as', 'j', 'f', 'ee', 'man', 'vas', 'ine', 'ix', 'de', 'w', 'vii', 'fin', 'ay', 'eh', 'eta', 'p', 'ok', 'xi', 'ja', 'm', 'su', 'for', 'aun', 'bus', 'it', 'q', 'lic', 'gil', 'sl', 'on', 've', 'en', 'si', 'té', 'xx', 'yo', 'iva', 'xvi', 'vos', 'us', 'iii', 'gen', 'cd', 'tú', 'mío', 'xix', 'con', 'new', 'ana', 'and', 'o', 'b', 'ex', 'k', 'com', 'h', 'á', 'et', 'ii', 'tv', 'mí', 'él', 'ti', 'muy', 'x', 'asi', 'the', 'mas', 'os', 'me', 'te', 'mi', 'tu', 'file', 're', 'viet']
keywords=[kw for kw in keywords if kw not in del_list]

#filtering keywords which appear less than 400 times in the EZLN corpus
EZLN_wc=pd.read_csv('wordlist_SE.csv', skiprows=[0, 1])
EZLN400=EZLN_wc[EZLN_wc['Frequency']>= 400]
EZLN400=EZLN400['Item'].to_list()

print(len(EZLN400))

keywords_EZLN=[kw for kw in EZLN400 if kw in keywords]
with open('EZLN400_kw.txt', 'w') as f:
    for kw in keywords_EZLN:
        f.write(f"{kw}\n")

#########################################################
#Clean corpus and add a reference code to potential SDKs#
#########################################################



#each record in group.json contains the texts belonging to the group concatenated together, as well as a code identifying the group/subcorpus
df=pd.read_json('final_df_JCDS.json', lines=True)

def remove_extra_spaces(text):
    return ' '.join(text.split())


def remove_non_alpha(text):
    return re.sub("[^A-Za-z0-9áéíóúüñÁÉÑÍÓ_@']+", ' ', text)


def lemmatizer(text):
    #text=text.lower()
    doc = nlp(text)
    txt=[word.lemma_ for word in doc if not word.is_stop]
    if len(txt)>2:
        return ' '.join(txt)


def process_text(text):
    text = text.lower()
    #text = remove_non_alpha(text)
    text = remove_extra_spaces(text)
    text = lemmatizer(text)
    return text

def replace_keywords(vec):
    
    orga=vec[0]
    text=vec[1]
    if orga=='EZLN':
        try:
            text_l=text.split(' ')
            for kw in keywords_EZLN:
                if kw in text_l:
                    text_l=list(map(lambda x: x.replace(kw, kw+'_'+orga), text_l))
            new_text=' '.join(text_l)
        except:
            print('ERROR', orga, text, type(text))
            new_text=' '
    else:
        new_text=text

    return new_text


def l_sentences(text):
    doc=nlp(text)
    sentences=[sent.text.strip() for sent in doc.sents if len(sent.text.strip().split())>2]
    return sentences


df['clean_text']=df['text'].apply(process_text)
df['SDK_text']=df[['organisation','clean_text']].apply(replace_keywords, axis=1)
df['SDK_text_sents']=df['SDK_text'].apply(l_sentences)


def remove_non_alpha_list(sents):
    return [re.sub("[^A-Za-z0-9áéíóúüñÁÉÑÍÓ_@']+", ' ', text) for text in sents]

def remove_non_alpha(text):
    return re.sub("[^A-Za-z0-9áéíóúüñÁÉÑÍÓ_@']+", ' ', text)

df['SDK_text_sents_clean']=df['SDK_text_sents'].apply(remove_non_alpha_list)
df['SDK_text_clean']=df['SDK_text'].apply(remove_non_alpha)


###########################
#Add stratification groups#
###########################


def joinx(c, o):
    c=c.replace(' ', '-')
    o=o.replace(' ', '-')
    o=o.replace('/', '')
    return '_'.join([c, o])

df['code']=df.apply(lambda x: joinx(x['country'], x['organisation']), axis=1)
strat_group=df.groupby(['country', 'organisation', 'code']).count().reset_index().sort_values(by='text', ascending=False).reset_index(drop=True)
standalone=strat_group[strat_group['text']>=50].code.to_list()
misc=strat_group[strat_group['text']<50].reset_index(drop=True)
misc_country=misc.groupby('country').sum().reset_index()
country_misc=misc_country[misc_country['text']>=50].country.to_list()
full_misc=misc_country[misc_country['text']<50].country.to_list()

def strat(code, country):
    if code in standalone:
        return code
    else:
        if country in country_misc:
            return country+'_misc'
        elif country in full_misc:
            return 'full_misc'
        else:
            print(code, 'ERROR')

df['strat_group']=df.apply(lambda x: strat(x['code'], x['country']), axis=1)


df.to_json('df_SDK_clean_2.json', orient='records', lines=True)