# -*- coding: utf-8 -*-
"""Rebuild the paper body into the requested ten-section structure."""
import os, re, sys
SC = '/tmp/claude-0/-home-user-Twinshield/28c50788-17b6-5fcf-b899-a73664e0ef5c/scratchpad'
sys.path.insert(0, SC + '/sc_scripts')
sys.path.insert(0, SC + '/paper')
from paperlib import Paper, shell
import content_a as A, content_b as B, content_c as C, content_d as D
import tables as T

DOC = sys.argv[1]
P = Paper(DOC)
d = P.d

for k, f in [('SEC', 'I.  INTRODUCTION'), ('SUB', 'A.  SYSTEM OVERVIEW'),
             ('BODY', 'Fig. 1 shows the architecture'), ('TLABEL', 'TABLE I'),
             ('TCAP', 'Summary of representative CAN'), ('FCAP', 'FIGURE 2.'),
             ('EQ', 'z_frame(t) = max_b')]:
    P.grab(k, f)

# blocks to carry over verbatim
KEEP_TABLES = [i for i, b in enumerate(d.blocks) if b.startswith('<w:tbl')][:2]
FIGS = {}
for key, rid in [('fig1', 'rId17'), ('fig2', 'rId9'), ('fig3', 'rId10'),
                 ('fig4', 'rId11'), ('fig5', 'rId16')]:
    hit = [i for i, b in enumerate(d.blocks) if f'r:embed="{rid}"' in b]
    assert len(hit) == 1, f"{key}: {len(hit)} blocks carry {rid}"
    FIGS[key] = d.blocks[hit[0]]
# a figure the template does not already contain needs its own media part and rel
def add_media(src, name):
    import shutil
    shutil.copy(src, os.path.join(os.path.dirname(DOC), 'media', name))
    rp = os.path.join(os.path.dirname(DOC), '_rels', 'document.xml.rels')
    rels = open(rp, encoding='utf-8').read()
    rid = 'rId%d' % (max(int(n) for n in re.findall(r'Id="rId(\d+)"', rels)) + 1)
    rels = rels.replace('</Relationships>',
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/image" Target="media/{name}"/>'
        '</Relationships>')
    open(rp, 'w', encoding='utf-8').write(rels)
    return rid

BRK2 = [b for i, b in enumerate(d.blocks)
        if b.startswith('<w:p') and '<w:sectPr' in b and 'w:num="2"' in b and i < 60]
BRK1 = [b for i, b in enumerate(d.blocks)
        if b.startswith('<w:p') and '<w:sectPr' in b and 'w:num="2"' not in b and 11 < i < 60]
assert BRK2 and BRK1, "could not find the one-column island's section breaks"
KEPT = [d.blocks[i] for i in KEEP_TABLES]

# Figure 6 is new: clone Figure 2's paragraph and point it at a fresh image part
from PIL import Image
OP = os.path.join('/home/user/Twinshield', 'report', 'figures',
                  'fig_operating_point.png')
_rid = add_media(OP, 'image90.png')
_im = Image.open(OP)
_cx = 2926080
_cy = int(_cx * _im.height / _im.width)
_f6 = FIGS['fig2']
_f6 = re.sub(r'r:embed="rId\d+"', f'r:embed="{_rid}"', _f6)
_f6 = re.sub(r'cx="\d+" cy="\d+"', f'cx="{_cx}" cy="{_cy}"', _f6)
FIGS['fig6'] = _f6

# ---------------------------------------------------------------- assemble
out = []
def emit(items):
    for kind, payload in items:
        if kind == 'sec':
            out.append(P.p('SEC', payload))
        elif kind == 'sub':
            out.append(P.p('SUB', payload))
        elif kind == 'body':
            out.append(P.p('BODY', payload))
        elif kind == 'eq':
            out.append(shell(P.t['EQ'], payload))
        elif kind == 'tlabel':
            out.append(P.p('TLABEL', payload))
        elif kind == 'tcap':
            out.append(P.p('TCAP', payload))
        elif kind == 'fcap':
            out.append(P.p('FCAP', payload))
        elif kind == 'keeptable':
            out.append(KEPT[payload])
            out.append(P.p('BODY', ''))
        elif kind == 'table':
            out.append(TABLES[payload])
            out.append(P.p('BODY', ''))
        elif kind == 'fig1':
            out.extend([BRK2[0], FIGS['fig1'],
                        P.p('FCAP',
                            "FIGURE 1.  TwinShield-Fusion. Raw CAN traffic is "
                            "preprocessed once and read by two branches. The digital "
                            "twin predicts the next payload of each arbitration "
                            "identifier from normal traffic alone, and its prediction "
                            "residual is standardised per identifier and per byte and "
                            "thresholded at a calibrated false-positive rate (Sections "
                            "V-C and V-D). The supervised classifier, a CNN-BiLSTM in "
                            "this work, names the attacks it was trained on (Section "
                            "V-E). The residual anomaly gate combines the two scores, "
                            "which is what lets an attack with no label of its own "
                            "still raise an alarm."),
                        BRK1[0]])
        elif kind in FIGS:
            out.append(FIGS[kind])
        else:
            raise SystemExit(f"unknown kind {kind!r}")

TABLES = {
    'DATASET_CHARS': T.dataset_chars(),
    'GEM_CLOSED': T.gem_closed(),
    'GEM_ABL': T.gem_abl(),
    'GEM_ZD': T.gem_zd(),
    # placeholders stay valid paragraphs until the runs that fill them finish
    'CIC_CLOSED': T.cic_closed(),
    'CIC_ZD': T.cic_zd(),
    'CROSS': T.cross(),
}

emit(A.SEC2); emit(B.SEC3); emit(B.SEC4)
emit(C.SEC5); emit(C.SEC6)
emit(D.SEC7); emit(D.SEC8); emit(D.SEC9)

# ------------------------------------------------- abstract and index terms
abs_i = [i for i, b in enumerate(d.blocks)
         if b.startswith('<w:p') and 'ABSTRACT' in d.text(i)][0]
idx_i = [i for i, b in enumerate(d.blocks)
         if b.startswith('<w:p') and 'INDEX TERMS' in d.text(i)][0]
d.blocks[abs_i] = shell(d.blocks[abs_i], [('ABSTRACT ', True), (A.ABSTRACT, False)])
d.blocks[idx_i] = shell(d.blocks[idx_i], [('INDEX TERMS ', True), (A.INDEX, False)])

# ----------------------------------------------------------- author block
AUTHORS = [
    ("FIRST A. AUTHOR\u00b9, SECOND B. AUTHOR\u00b2, and THIRD C. AUTHOR\u00b3",
     "RAMYA NAKSHATHRA\u00b9, AGILAN\u00b9, and S. ANUBHA PEARLINE\u00b9"),
    ("\u00b9Department of Computer Science and Engineering, [Institution], [City], [Country]",
     "\u00b9School of Computer Science and Engineering, Vellore Institute of "
     "Technology, Chennai 600127, India"),
    ("\u00b2Department of Computer Science and Engineering, [Institution], [City], [Country]",
     ""),
    ("\u00b3Department of Computer Science and Engineering, [Institution], [City], [Country]",
     ""),
    ("Corresponding author: First A. Author (e-mail: author@institution.edu).",
     "Corresponding author: S. Anubha Pearline (e-mail: [corresponding author e-mail])."),
]
for old, new in AUTHORS:
    hit = [i for i, b in enumerate(d.blocks)
           if b.startswith('<w:p') and d.text(i).strip() == old]
    if not hit:
        continue
    if new:
        d.blocks[hit[0]] = shell(d.blocks[hit[0]], new)
    else:
        d.blocks[hit[0]] = ''

bios = [i for i, b in enumerate(d.blocks)
        if b.startswith('<w:p') and 'received the B.E. degree' in d.text(i)]
BIO = [
 "RAMYA NAKSHATHRA is currently pursuing the B.Tech. degree in computer science and "
 "engineering with the School of Computer Science and Engineering, Vellore Institute "
 "of Technology, Chennai, India. Her research interests include automotive "
 "cybersecurity, intrusion detection and deep learning for time-series data.",
 "AGILAN is currently pursuing the B.Tech. degree in computer science and "
 "engineering with the School of Computer Science and Engineering, Vellore Institute "
 "of Technology, Chennai, India. His research interests include in-vehicle network "
 "security, anomaly detection and explainable machine learning.",
 "S. ANUBHA PEARLINE is an Assistant Professor with the School of Computer Science "
 "and Engineering, Vellore Institute of Technology, Chennai, India. [Degrees, "
 "awarding institutions and research interests to be completed by the author.]",
]
for i, text in zip(bios, BIO):
    d.blocks[i] = shell(d.blocks[i], text)

# ------------------------------------------------- title and funding note
ttl = [i for i, b in enumerate(d.blocks)
       if b.startswith('<w:p') and d.text(i).strip().startswith('TwinShield-Fusion:')]
if ttl:
    d.blocks[ttl[0]] = shell(d.blocks[ttl[0]],
        "TwinShield-Fusion: Digital-Twin Residual Gating for Unseen Cyber-Physical "
        "Attacks on Electric Vehicle CAN Buses")
note = [i for i, b in enumerate(d.blocks)
        if b.startswith('<w:p') and 'undergraduate research project' in d.text(i)]
if note:
    d.blocks[note[0]] = shell(d.blocks[note[0]],
        "This work was carried out as part of an undergraduate research project. The "
        "GEM-CAN dataset is distributed under CC-BY-NC-3.0 and was obtained from "
        "Zenodo (10.5281/zenodo.17834776). The CICIoV2024 dataset is distributed by "
        "the Canadian Institute for Cybersecurity, University of New Brunswick.")

# ------------------------------------------------------------- splice body
start = [i for i, b in enumerate(d.blocks)
         if b.startswith('<w:p') and d.text(i).strip() == 'I.  INTRODUCTION'][0]
end = [i for i, b in enumerate(d.blocks)
       if b.startswith('<w:p') and d.text(i).strip() == 'REFERENCES'][0]
print(f"replacing blocks {start}..{end - 1} with {len(out)} new blocks")
d.blocks[start:end] = out
d.save()
print("body rebuilt")
