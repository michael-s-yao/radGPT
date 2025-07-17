# Evaluating Acute Image Ordering for Real-World Patient Cases via Language Model Alignment with Radiological Guidelines

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-obastani%40seas.upenn.edu-blue)](mailto:obastani@seas.upenn.edu)
[![DOI](https://zenodo.org/badge/819630412.svg)](https://doi.org/10.5281/zenodo.16055680)

Diagnostic imaging studies are increasingly important in the management of acutely presenting patients. However, ordering appropriate imaging studies in the emergency department is a challenging task with a high degree of variability among healthcare providers. To address this issue, recent work has investigated whether generative AI and large language models can be leveraged to recommend diagnostic imaging studies in accordance with evidence-based medical guidelines. However, it remains challenging to ensure that these tools can provide recommendations that correctly align with medical guidelines, especially given the limited diagnostic information available in acute care settings. In this study, we introduce a framework to intelligently leverage language models by recommending imaging studies for patient cases that align with the American College of Radiology's Appropriateness Criteria, a set of evidence-based guidelines. To power our experiments, we introduce *RadCases*, a dataset of over 1500 annotated case summaries reflecting common patient presentations, and apply our framework to enable state-of-the-art language models to reason about appropriate imaging choices. Using our framework, state-of-the-art language models achieve accuracy comparable to clinicians in ordering imaging studies. Furthermore, we demonstrate that our language model-based pipeline can be used as an intelligent assistant by clinicians to support image ordering workflows and improve the accuracy of acute image ordering according to the American College of Radiology's Appropriateness Criteria. Our work demonstrates and validates a strategy to leverage AI-based software to improve trustworthy clinical decision-making in alignment with expert evidence-based guidelines.

## Installation

To install and run our code, first clone the `radGPT` repository.

```
git clone https://github.com/michael-s-yao/radGPT
cd radGPT
```

Next, create a conda environment and install the relevant dependencies.

```
conda env create -f environment.yml
conda activate radgpt
```

After successful setup, you can run our code as

```
python main.py --help
```

## Contact

Questions and comments are welcome. Suggestions can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Osbert Bastani](mailto:obastani@seas.upenn.edu)

## Citation

When available, relevant citation information will be added in a future commit.

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
