# -*- coding: utf-8 -*-
"""
knowledge about ECG arrhythmia
"""


__all__ = [
    "AF", "I_AVB", "LBBB", "RBBB", "PAC", "PVC",
    "STD", "STE",
]


AF = {
    "url": [
        "https://litfl.com/atrial-fibrillation-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrial_fibrillation#Screening",
    ],
    "knowledge": [
        "Irregularly irregular rhythm",
        "No P waves",
        "Absence of an isoelectric baseline",
        "Variable ventricular rate",
        "QRS complexes usually < 120 ms unless pre-existing bundle branch block, accessory pathway, or rate related aberrant conduction",
        "Fibrillatory waves (f-wave) may be present and can be either fine (amplitude < 0.5mm) or coarse (amplitude >0.5mm)",
        "Fibrillatory waves (f-wave) may mimic P waves leading to misdiagnosis",
    ],
}

I_AVB = {
    "url": [
        "https://litfl.com/first-degree-heart-block-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrioventricular_block#First-degree_Atrioventricular_Block"
    ],
    "knowledge": [
        "PR interval > 200ms",
        "Marked’ first degree block if PR interval > 300ms",
        "P waves might be buried in the preceding T wave",
        "There are no dropped, or skipped, beats"
    ],
}

LBBB = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}

RBBB = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}

PAC = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}

PVC = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}

STD = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}

STE = {
    "url": [
        "",
    ],
    "knowledge": [
        ""
    ],
}
