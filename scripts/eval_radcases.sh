#!/usr/bin/bash

for ds in "synthetic" "medbullets" "jama_cc" "nejm" "mimic_iv" "wikitext" "MaartenGr/arxiv_nlp" "bigbio/med_qa" "pubmed"; do
  for metric in "perplexity" "tok" "similarity"; do
    python eval_radcases.py -d $ds -m $metric
  done
done
