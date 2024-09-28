# Evidence Is All You Need: Ordering Imaging Studies via Language Model Alignment with the ACR Appropriateness Criteria

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-obastani%40seas.upenn.edu-blue)](mailto:obastani@seas.upenn.edu)

Diagnostic imaging studies are an increasingly important component of the workup and management of acutely presenting patients. However, ordering appropriate imaging studies according to evidence-based medical guidelines is a challenging task with a high degree of variability between healthcare providers. To address this issue, recent work has investigated if generative AI and large language models can be leveraged to help clinicians order relevant imaging studies for patients. However, it is challenging to ensure that these tools are correctly aligned with medical guidelines, such as the American College of Radiology's Appropriateness Criteria (ACR AC). In this study, we introduce a framework to intelligently leverage language models by recommending imaging studies for patient cases that are aligned with evidence-based guidelines. We make available a novel dataset of patient "one-liner" scenarios to power our experiments, and optimize state-of-the-art language models to achieve an accuracy on par with clinicians in image ordering. Finally, we demonstrate that our language model-based pipeline can be used as intelligent assistants by clinicians to support image ordering workflows and improve the accuracy of imaging study ordering according to the ACR AC. Our work demonstrates and validates a strategy to leverage AI-based software to improve trustworthy clinical decision making in alignment with expert evidence-based guidelines.

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
