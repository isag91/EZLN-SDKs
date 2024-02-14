# EZLN-SDKs

This is the repository accompanying the paper "The Zapatista Semantic Struggle: Analysing the Linguistic Innovation of the EZLN with Semantic Difference Keywords (SDKs)".

The corpus used for the case study was assembled from three sources:

1. **The CeDeMa archive (Centro de Documentacion de los movimientos armados)**\
URL: https://cedema.org/digital_items \
All documents issued by a movement, written in Spanish, and dated from 1953 onward were selected. Documents which were available in formats other than plain text where converted to plain text. PDF files which needed to be OCRed were OCRed by using Tesseract and Google Vision.[^2]

2. **The archive of the 26th of July Movement (the leading organisation of the Cuban Revolution) and the Castro regime**\
URL: http://www.fidelcastro.cu/es/biblioteca/documentos/coleccion/todas

3. **The archive of the EZLN (Zapatista Army of National Liberation)**\
URL: https://enlacezapatista.ezln.org.mx/category/comunicado/ \
All the original Spanish texts were selected. 

Although this data is publicly available, we do not have permission to redistribute it. However, **the embeddings of the Word2Vec models are available here**: TBC.

Here is a short overview of the files in this directory.

| Filename    | Content |
| -------- | ------- |
| keywords_SE.csv | This file was obtained by comparing the entire corpus with the general Spanish language corpus esTenTen on Sketch Engine. The keyness scores are computed with the simple maths keyness metric.[^1]   |
| peprocessing_JCADS.py  | The script for the preprocessing steps of the raw text data and the statistical keyness data obtained from Sketch Engine. The text data is lemmatized and split into sentences. Statistical keywords which are not nouns, are stop words or do not exist are discarded. In addition, they are filtered according to a keyness threshold, and frequency values in the target and reference corpora. Then, the contextual reference is appended to words selected as potential SDKs. Finally, the corpus is separated in different stratification groups for bootstrap sampling.|
| EZLN400_kw.txt | The list of potential SDKs obtained from preprocessing_JCADS.py|
| train_JCADS.py | The script for training the Word2Vec models (including hyperparameters).|

[^1]: Adam Kilgarriff. 2009. Simple maths for keywords. In Proc. Corpus Linguistics, volume 6.
[^2]: Isabelle Gribomont, "OCR with Google Vision API and Tesseract," Programming Historian 12 (2023), https://doi.org/10.46430/phen0109.

