import pandas as pd
import multiprocessing
import re

from gensim.models import Word2Vec
cores = multiprocessing.cpu_count()

df=pd.read_json('df_SDK_clean_2.json', lines=True)


df_F=df[['SDK_text_clean', 'SDK_text_sents_clean', 'strat_group']]
df_F=df_F[df_F['SDK_text_clean'].notnull()]


def train_model(input_length):
	
	if input_length=='documents':
		print(input_length)
		documents = [row.split() for row in strat['SDK_text_clean'] if len(row.split())>2]
	elif input_length=='sentences':
		print(input_length)
		strat_ex=strat[['SDK_text_sents_clean', 'strat_group']].explode('SDK_text_sents_clean')
		strat_ex = strat_ex[strat_ex['SDK_text_sents_clean'].notnull()]
		documents = [row.split() for row in strat_ex['SDK_text_sents_clean'] if len(row.split())>2]
	else:
		print(input_length, 'Wrong input_length')
	

	model = Word2Vec(documents,
	                 window=5,
	                 min_count=20,
	                 sg=1,
	                 negative=20,
	                 workers=cores-1,

	                 vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007
                     )
	model.build_vocab(documents)
	model.train(documents, total_examples=model.corpus_count, epochs=30)
	#model.init_sims(replace=True)
	model.save('models/3_{}_{}_old_params.model'.format(i, input_length))

for i in range(20):
	print(i)
	try:
		strat=df_F.groupby('strat_group').apply(lambda x: x.sample(frac=1, replace=True))
		#train_model('documents')
		train_model('sentences')
	except:
		print('OOPS')







