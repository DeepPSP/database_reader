# -*- coding: utf-8 -*-
"""
knowledge about ECG arrhythmia
"""


__all__ = [
    "AF",
    "I_AVB",
    "LBBB", "RBBB",
    "PAC", "PJC", "PVC", "SPB",
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
        "https://litfl.com/premature-atrial-complex-pac/",
        "https://en.wikipedia.org/wiki/Premature_atrial_contraction",
    ],
    "knowledge": [
        "An abnormal (non-sinus) P wave is followed by a QRS complex",
        "P wave typically has a different morphology and axis to the sinus P waves",
        "Abnormal P wave may be hidden in the preceding T wave, producing a “peaked” or “camel hump” appearance",
        # to add more
    ],
}

PJC = {
    "url": [
        "https://litfl.com/premature-junctional-complex-pjc/",
        "https://en.wikipedia.org/wiki/Premature_junctional_contraction",
    ],
    "knowledge": [
        "Narrow QRS complex, either (1) without a preceding P wave or (2) with a retrograde P wave which may appear before, during, or after the QRS complex. If before, there is a short PR interval of < 120 ms and the  “retrograde” P waves are usually inverted in leads II, III and aVF",
        "Occurs sooner than would be expected for the next sinus impulse",
        "Followed by a compensatory pause",
    ],
}

PVC = {
    "url": [
        "https://litfl.com/premature-ventricular-complex-pvc-ecg-library/",
        "https://en.wikipedia.org/wiki/Premature_ventricular_contraction",
    ],
    "knowledge": [
        "Broad QRS complex (≥ 120 ms) with abnormal morphology",
        "Premature — i.e. occurs earlier than would be expected for the next sinus impulse",
        "Discordant ST segment and T wave changes",
        "Usually followed by a full compensatory pause",
        "Retrograde capture of the atria may or may not occur",
    ],
}

SPB = {
    "url": [
        "https://en.wikipedia.org/wiki/Premature_atrial_contraction#Supraventricular_extrasystole",
    ],
    "knowledge": PAC["knowledge"] + PJC["knowledge"],
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
