# -*- coding: utf-8 -*-
"""Table bodies for the restructured paper. Numbers come from results/."""
import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-Twinshield/28c50788-17b6-5fcf-b899-a73664e0ef5c/scratchpad/sc_scripts')
from mktable import table

W3 = [1800, 1450, 1450]
W3b = [2300, 1200, 1200]
W4 = [1700, 1050, 1000, 950]

def dataset_chars():
    return table(
        ["Property", "GEM-CAN", "CICIoV2024"],
        [["Vehicle", "Polaris GEM e6\n(electric, drive-by-wire)", "2019 Ford\n(production)"],
         ["Setting", "Closed rural test track", "Road capture"],
         ["Bus speed", "250 kbit/s", "not stated"],
         ["Frames used", "143,444", "259,977"],
         ["Classes", "4", "6 (grouped to 4)"],
         ["Timestamp", "recorded", "absent; synthesised"],
         ["Data length code", "recorded", "absent; fixed at 8"],
         ["Volumetric attack", "DoS on ID 00000000", "DoS on one ID"],
         ["Spoofing attacks", "brake, steering", "gas, RPM, speed, steering"],
         ["ID-only accuracy", "98.90%", "100.00% (macro-F1 100.00%)"]],
        W3)

def gem_closed():
    return table(
        ["Model", "CAN ID included", "CAN ID excluded"],
        [["CNN", "99.68", "99.14"],
         ["BiLSTM", "99.48", "98.95"],
         ["CNN-BiLSTM", "99.79", "99.14"],
         ["GRU Digital Twin + CNN-BiLSTM", "99.89", "99.79"],
         ["TwinShield-Fusion", "99.79", "99.79"]],
        W3b, bold_rows=(4,))

def gem_abl():
    return table(
        ["Configuration", "CAN ID incl.", "CAN ID excl."],
        [["CNN-BiLSTM", "99.79", "99.14"],
         ["+ digital twin (scalar)", "99.58", "99.65"],
         ["+ residual vector", "99.89", "99.79"],
         ["+ adaptive fusion", "99.79", "99.79"]],
        W3b)

def gem_zd():
    return table(
        ["Model", "Detected", "False alarms", "Known macro-F1 (%)"],
        [["CNN", "0.84%", "0.03%", "66.67"],
         ["BiLSTM", "0.00%", "0.08%", "76.07"],
         ["CNN-BiLSTM", "25.21%", "0.31%", "77.87"],
         ["TwinShield-Fusion", "100.00%", "2.26%", "78.29"]],
        W4, bold_rows=(3,))


def cic_closed():
    return table(
        ["Model", "CAN ID included", "CAN ID excluded"],
        [["CNN", "100.00", "99.89"],
         ["BiLSTM", "100.00", "99.85"],
         ["CNN-BiLSTM", "100.00", "99.92"],
         ["GRU Digital Twin + CNN-BiLSTM", "100.00", "100.00"],
         ["TwinShield-Fusion", "100.00", "99.99"]],
        W3b, bold_rows=(3, 4))


def cic_zd():
    return table(
        ["Quarantined class", "TwinShield-Fusion", "Best baseline", "False alarms"],
        [["DoS", "100.00%", "0.00%", "0.37%"],
         ["Powertrain spoofing", "100.00%", "35.71%", "0.37%"],
         ["Steering spoofing", "100.00%", "0.16%", "0.37%"]],
        [1750, 1200, 1050, 1000], bold_rows=(0, 1, 2))


def cross():
    return table(
        ["Dataset / vehicle", "Quarantined class", "Detected", "Best baseline"],
        [["GEM-CAN, Polaris GEM e6", "brake tampering", "100.00%", "25.21%"],
         ["CICIoV2024, 2019 Ford", "DoS", "100.00%", "0.00%"],
         ["CICIoV2024, 2019 Ford", "powertrain spoofing", "100.00%", "35.71%"],
         ["CICIoV2024, 2019 Ford", "steering spoofing", "100.00%", "0.16%"]],
        [1750, 1300, 950, 1000], bold_rows=(0, 1, 2, 3))
