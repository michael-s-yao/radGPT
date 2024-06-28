# Instructions for Data Preparation

Please find our instructions below on how to prepare and download each of the four benchmarking datasets introduced in our work. For additional questions, open a GitHub Issue or contact us [here](mailto:michael.yao@pennmedicine.upenn.edu).

## Synthetic Dataset

The synthetically generated dataset of patient one-liners can be found [here](https://docs.google.com/spreadsheets/d/1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/edit?usp=sharing&gid=1839683815).

## Medbullets Dataset

Our dataset derived from the Medbullets dataset initially introduced by [Chen et al. (2024)](https://arxiv.org/abs/2402.18060v3) can be found [here](https://docs.google.com/spreadsheets/d/1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/edit?usp=sharing&gid=0). The dataset was constructed by taking the first sentence of each of the original question stems and using it as the input patient case.

## JAMA Clinical Challenges Dataset

Our dataset derived from the JAMA Clinical Challenges dataset initially introduced and described by [Chen et al. (2024)](https://arxiv.org/abs/2402.18060v3) can be found [here](https://docs.google.com/spreadsheets/d/1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/edit?usp=sharing&gid=226315523). Similar to the Medbullets dataset, this dataset was also constructed by taking the first sentence of each of the original question stems and using it as the input patient case.

Due to licensing constraints, we cannot release the input one-liners in the JAMA Clinical Challenges dataset publicly. In order to reproduce our experiments using this dataset, first follow the instructions [here](https://github.com/HanjieChen/ChallengeClinicalQA/blob/4a0c57544cb88e0d0f0adc8ba1060f3b847f3062/README.md) to download the JAMA Clinical Challenge dataset, and move the output `jama_raw.csv` file to the [parent directory](../..) of this repository. The ground truth labels of the corresponding one-liners can be found by hashing an input one-liner and using the hash-to-label lookup table [here](https://docs.google.com/spreadsheets/d/1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/edit?usp=sharing&gid=226315523), which is what is done in our implementation.

## MIMIC-IV Dataset

Due to licensing constraints, we cannot release the input one-liners in this dataset publicly, and instead make the hash representations of the one-liners and their corresponding labels available [here](https://docs.google.com/spreadsheets/d/1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/edit?usp=sharing&gid=1839683815) (exactly as was done for the JAMA Clinical Challenges Dataset described above). To reproduce our experiments involving this dataset, please follow the instructions [here](https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel) to download the table of discharge notes `discharge.csv.gz` to the [`radgpt/data`](.) of this repository. Experiments can then be performed using our codebase once this step is completed.