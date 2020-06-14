# -*- coding: utf-8 -*-
"""
knowledge about ECG arrhythmia, and corresponding Dx maps

Standard_12Leads_ECG:
---------------------
    Inferior leads: II, III, aVF
    Lateral leads: I, aVL, V5-6
    Septal leads: V1, aVR
    Anterior leads: V2-4
    -----------------------------------
    Chest (precordial) leads: V1-6
    Limb leads: I, II, III, aVF, aVR, aVL
"""
from io import StringIO
import pandas as pd


__all__ = [
    "AF",
    "I_AVB",
    "LBBB", "RBBB",
    "PAC", "PJC", "PVC", "SPB",
    "STD", "STE",
    "Dx_map",
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
        "https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Left_bundle_branch_block",
    ],
    "knowledge": [
        "Heart rhythm must be supraventricular",
        "QRS duration of > 120 ms",
        "Lead V1: Dominant S wave, with QS or rS complex",
        "Lateral leads: M-shaped, or notched, or broad monophasic R wave or RS complex; absence of Q waves (small Q waves are still allowed in aVL)",
        "Chest (precordial) leads: poor R wave progression",
        "Left precordial leads (V5-6): prolonged R wave peak time > 60ms",
        "ST segments and T waves always go in the opposite direction to the main vector of the QRS complex",
    ],
}

RBBB = {
    "url": [
        "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Right_bundle_branch_block",
    ],
    "knowledge": [
        "Broad QRS > 100 ms (incomplete block) or > 120 ms (complete block)",
        "Leads V1-3: RSR’ pattern (‘M-shaped’ QRS complex); sometimes a broad monophasic R wave or a qR complex in V1",
        "Lateral leads: wide, slurred S wave",
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


# ref. https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv
# NOTE that 'SNR' is the 'Normal' rhythm
Dx_map = pd.read_csv(StringIO("""dx,SNOMED code,Abbreviation
1st degree av block,270492004,IAVB
2nd degree av block,195042002,IIAVB
accelerated idioventricular rhythm,61277005,AIVR
accelerated junctional rhythm,426664006,AJR
acute myocardial ischemia,413444003,AMIs
anterior ischemia,426434006,AnMIs
anterior myocardial infarction,54329005,AnMI
atrial fibrillation,164889003,AF
atrial fibrillation and flutter,195080001,AFAFL
atrial flutter,164890007,AFL
atrial hypertrophy,195126007,AH
atrial pacing pattern,251268003,AP
atrial tachycardia,713422000,ATach
av block,233917008,AVB
blocked premature atrial contraction,251170000,BPAC
brady tachy syndrome,74615001,BTS
bradycardia,426627000,Brady
brugada syndrome,418818005,Brug
bundle branch block,6374002,BBB
chronic atrial fibrillation,426749004,CAF
chronic myocardial ischemia,413844008,CMIs
chronic rheumatic pericarditis,78069008,CRPC
complete heart block,27885002,CHB
complete right bundle branch block,713427006,CRBBB
decreased qt interval,77867006,SQT
diffuse intraventricular block,82226007,DIVB
early repolarization,428417006,ERe
ecg artefacts,251143007,ART
ectopic rhythm,29320008,ER
electrical alternans,423863005,EA
high t voltage,251259000,HTV
incomplete left bundle branch block,251120003,ILBBB
incomplete right bundle branch block,713426002,IRBBB
indeterminate cardiac axis,251200008,ICA
inferior ischaemia,425419005,IIs
inferior st segment depression,704997005,ISTD
isorhythmic dissociation,50799005,IAVD
junctional escape,426995002,JE
junctional premature complex,251164006,JPC
junctional tachycardia,426648003,JTach
lateral ischaemia,425623009,LIs
left anterior fascicular block,445118002,LAnFB
left atrial abnormality,253352002,LAA
left atrial enlargement,67741000119109,LAE
left atrial hypertrophy,446813000,LAH
left axis deviation,39732003,LAD
left bundle branch block,164909002,LBBB
left posterior fascicular block,445211001,LPFB
left ventricular hypertrophy,164873001,LVH
left ventricular strain,370365005,LVS
low qrs voltages,251146004,LQRSV
low qrs voltages in the limb leads,251147008,LQRSVLL
low qrs voltages in the precordial leads,251148003,LQRSP
mobitz type 2 second degree atrioventricular block,28189009,MoII
mobitz type i wenckebach atrioventricular block,54016002,MoI
multifocal atrial tachycardia,713423005,MATach
myocardial infarction,164865005,MI
myocardial ischemia,164861001,MIs
non-specific interatrial conduction block,65778007,NSIACB
nonspecific intraventricular conduction disorder,698252002,NSIVCB
nonspecific st t abnormality,428750005,NSSTTA
old myocardial infarction,164867002,OldMI
pacing rhythm,10370003,PR
paroxysmal supraventricular tachycardia,67198005,PSVT
partial atrioventricular block 2:1,164903001,PAVB21
premature atrial contraction,284470004,PAC
premature ventricular complexes,164884008,PVC
premature ventricular contractions,427172004,PVC
prolonged qt interval,111975006,LQT
qwave abnormal,164917005,QAb
r wave abnormal,164921003,RAb
right atrial abnormality,253339007,RAAb
right atrial hypertrophy,446358003,RAH
right axis deviation,47665007,RAD
right bundle branch block,59118001,RBBB
right ventricular hypertrophy,89792004,RVH
s t changes,55930002,STC
shortened pr interval,49578007,SPRI
sinus arrhythmia,427393009,SA
sinus bradycardia,426177001,SB
sinus rhythm,426783006,SNR
sinus tachycardia,427084000,STach
st depression,429622005,STD
st elevation,164931005,STE
st interval abnormal,164930006,STIAb
supraventricular premature beats,63593006,SVPB
supraventricular tachycardia,426761007,SVT
suspect arm ecg leads reversed,251139008,ALR
t wave abnormal,164934002,TAb
t wave inversion,59931005,TInv
tall u wave,251242005,UTall
u wave abnormal,164937009,UAb
ventricular bigeminy,11157007,VBig
ventricular ectopic beats,17338001,VEB
ventricular escape beat,75532003,VEsB
ventricular escape rhythm,81898007,VEsR
ventricular fibrillation,164896001,VF
ventricular flutter,111288001,VFL
ventricular hypertrophy,266249003,VH
ventricular pacing pattern,251266004,VPP
ventricular pre excitation,195060002,VPEx
ventricular tachycardia,164895002,VTach
ventricular trigeminy,251180001,VTrig
wandering atrial pacemaker,195101003,WAP
wolff parkinson white pattern,74390002,WPW"""))
