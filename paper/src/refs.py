# -*- coding: utf-8 -*-
"""Rebuild the reference list to match the restructured text's citations."""
import re, sys
sys.path.insert(0, '/tmp/claude-0/-home-user-Twinshield/28c50788-17b6-5fcf-b899-a73664e0ef5c/scratchpad/sc_scripts')
from docutil import Doc
from paperlib import shell

# citation numbers used while drafting -> final numbers (1-14 unchanged)
REMAP = {17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
         24: 22, 25: 23, 26: 24, 27: 25, 28: 26, 29: 27}

# final number -> which existing entry to reuse (1-based), or new text
REUSE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 12, 7: 7, 8: 8, 9: 6,
         13: 13, 14: 14, 16: 18, 17: 15, 18: 16, 19: 17, 20: 19,
         21: 20, 22: 21, 24: 23, 25: 24, 26: 25, 27: 26}
NEW = {
    10: "R. Rai, J. Grover, P. Sharma, and A. Pareek, “Securing the CAN bus using "
        "deep learning for intrusion detection in vehicles,” Sci. Rep., vol. 15, "
        "art. 13820, 2025.",
    11: "X. Wang, J. Zhao, P. Liu, N. Yao, and Z. Xu, “CAN bus intrusion detection "
        "based on deep learning with data augmentation for connected autonomous "
        "vehicles,” IEEE Trans. Veh. Technol., vol. 75, no. 2, pp. 2253–2266, 2026.",
    12: "Upstream Security, “2022 global automotive cybersecurity report,” "
        "Upstream Security Ltd., 2022.",
    15: "B. Xu, Q. Li, F. Cao, and Y. Gao, “A Mamba-based lightweight and dynamically "
        "adaptive intrusion detection framework for in-vehicle CAN bus,” Internet "
        "of Things, vol. 37, art. 101918, 2026.",
    23: "Canadian Institute for Cybersecurity, “CICIoV2024: Internet of Vehicles "
        "dataset,” Univ. New Brunswick, Fredericton, NB, Canada, 2024. [Online]. "
        "Available: https://www.unb.ca/cic/datasets/iov-dataset-2024.html",
}


def run(path):
    d = Doc(path)
    ref_i = [i for i, b in enumerate(d.blocks)
             if b.startswith('<w:p') and d.text(i).strip() == 'REFERENCES'][0]
    bio_i = [i for i, b in enumerate(d.blocks)
             if b.startswith('<w:p') and 'received the B.E. degree' in d.text(i)][0]

    old = []
    for i in range(ref_i + 1, bio_i):
        if d.blocks[i].startswith('<w:p'):
            t = d.text(i).strip()
            if re.match(r'^\[\d+\]', t):
                old.append((i, t))
    assert len(old) >= 26, f"only {len(old)} reference entries found"
    tpl = d.blocks[old[0][0]]
    body = {}
    for _, t in old:
        n = int(re.match(r'^\[(\d+)\]', t).group(1))
        body[n] = re.sub(r'^\[\d+\]\s*', '', t)

    entries = []
    for n in range(1, 28):
        if n in NEW:
            text = NEW[n]
        else:
            src = REUSE.get(n)
            assert src in body, f"reference {n}: no source entry {src}"
            text = body[src]
        entries.append(shell(tpl, f"[{n}]  {text}"))

    d.blocks[old[0][0]:old[-1][0] + 1] = entries

    # renumber the in-text citations in one pass
    hits = [0]
    def fix(m):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(',')]
        out = []
        for p in parts:
            if p.isdigit():
                v = int(p)
                out.append(str(REMAP.get(v, v)))
                hits[0] += 1
            else:
                out.append(p)
        return '[' + ', '.join(out) + ']'
    for i, b in enumerate(d.blocks):
        if b.startswith('<w:p') and i < old[0][0]:
            d.blocks[i] = re.sub(r'\[([\d,\s]+)\]', fix, b)
    d.save()
    print(f"reference list rebuilt with {len(entries)} entries; "
          f"{hits[0]} in-text citation numbers remapped")


if __name__ == '__main__':
    run(sys.argv[1])
