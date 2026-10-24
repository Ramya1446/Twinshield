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
         ["ID-only accuracy", "98.9%", "100.0% (macro-F1 1.000)"]],
        W3)

def gem_closed():
    return table(
        ["Model", "CAN ID included", "CAN ID excluded"],
        [["CNN", "0.9968", "0.9914"],
         ["BiLSTM", "0.9948", "0.9895"],
         ["CNN-BiLSTM", "0.9979", "0.9914"],
         ["GRU Digital Twin + CNN-BiLSTM", "0.9989", "0.9979"],
         ["TwinShield-Fusion", "0.9979", "0.9979"]],
        W3b, bold_rows=(4,))

def gem_abl():
    return table(
        ["Configuration", "CAN ID incl.", "CAN ID excl."],
        [["CNN-BiLSTM", "0.9979", "0.9914"],
         ["+ digital twin (scalar)", "0.9958", "0.9965"],
         ["+ residual vector", "0.9989", "0.9979"],
         ["+ adaptive fusion", "0.9979", "0.9979"]],
        W3b)

def gem_zd():
    return table(
        ["Model", "Detected", "False alarms", "Known macro-F1"],
        [["CNN", "0.84%", "0.03%", "0.6667"],
         ["BiLSTM", "0.00%", "0.08%", "0.7607"],
         ["CNN-BiLSTM", "25.21%", "0.31%", "0.7787"],
         ["TwinShield-Fusion", "100.00%", "2.26%", "0.7829"]],
        W4, bold_rows=(3,))


def cic_closed():
    return table(
        ["Model", "CAN ID included", "CAN ID excluded"],
        [["CNN", "1.0000", "0.9989"],
         ["BiLSTM", "1.0000", "0.9985"],
         ["CNN-BiLSTM", "1.0000", "0.9992"],
         ["GRU Digital Twin + CNN-BiLSTM", "1.0000", "1.0000"],
         ["TwinShield-Fusion", "1.0000", "0.9999"]],
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
