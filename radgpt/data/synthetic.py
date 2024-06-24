"""
Reads the raw synthetic LLM-generated toy dataset.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Links to OpenAI ChatGPT Conversations:
    [a] Breast Prompts:
        https://chatgpt.com/share/625436fd-87dd-4be3-afd0-b59529d816ff
    [b] Cardiac Prompts:
        https://chatgpt.com/share/2d52c0b2-7bcd-4cce-ae14-3af033a3b225
    [c] Gastrointestinal Prompts:
        https://chatgpt.com/share/5cd86b6d-31dd-4eb1-8aa8-a22b955ddeac
    [d] Gyn and OB Prompts:
        https://chatgpt.com/share/b538a30a-9b53-414a-8644-a9a247e065d9
    [e] Musculoskeletal Prompts:
        https://chatgpt.com/share/9bd5164e-a923-4cfb-9cdb-64cb5384c531
    [f] Neurologic Prompts:
        https://chatgpt.com/share/5be73ee7-ed4a-4bb7-9923-c331d8f5023c
    [g] Pediatric Prompts:
        https://chatgpt.com/share/5913fb17-dee0-473c-835d-b0a05f78f456
    [h] Polytrauma Prompts:
        https://chatgpt.com/share/df0d95f7-1e61-4ae4-87be-fcf58960685e
    [j] Thoracic Prompts:
        https://chatgpt.com/share/f422a439-4a1b-4853-be32-d9e52efc7e56
    [k] Urologic Prompts:
        https://chatgpt.com/share/20dd9c03-7a40-484c-86c5-7f82695effa7
    [m] Vascular Prompts:
        https://chatgpt.com/share/289ac958-3f1f-46d8-9868-73360763ded5

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import pandas as pd


def read_synthetic_dataset(
    dataset_url: str = (
        "https://docs.google.com/spreadsheets/d/"
        "1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs/"
        "export?gid=1839683815&format=csv"
    )
) -> np.ndarray:
    """
    Returns the synthetic LLM-generated dataset of patient one-liners.
    Input:
        dataset_url: The URL to the dataset.
    Returns:
        An array of all the patient one-liners in the synthetic dataset.
    """
    return pd.read_csv(dataset_url)["case_readable"].to_numpy()
