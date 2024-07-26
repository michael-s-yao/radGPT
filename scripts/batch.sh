#!/bin/bash

llm=$1
method=$2
metric=${3:-panel}
savedir=${4:-results}

for ds in "synthetic" "medbullets" "jama_cc" "mimic_iv" "nejm"; do
  echo $ds
  for seed in 42 43 44 45 46; do
    python main.py \
      -l $llm \
      -m $method \
      -d $ds \
      --savedir $savedir \
      --seed $seed \
      --by-$metric \
      --cot.reasoning_method default \
      --rag.retriever BM25Retriever \
      --rag.top_k 8 \
      --rag.corpus MedRAG/textbooks \
      --icl.retriever MPNetRetriever \
      --icl.num_examples 4
  done
done
