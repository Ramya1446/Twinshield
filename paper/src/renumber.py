# -*- coding: utf-8 -*-
"""Shift the section numerals down one: the abstract is not a numbered section."""
import re, sys
sys.path.insert(0, '/tmp/claude-0/-home-user-Twinshield/28c50788-17b6-5fcf-b899-a73664e0ef5c/scratchpad/sc_scripts')
from docutil import Doc

MAP = {"II": "I", "III": "II", "IV": "III", "V": "IV",
       "VI": "V", "VII": "VI", "VIII": "VII", "IX": "VIII"}
ROM = r"(?:IX|VIII|VII|VI|IV|V|III|II)"

d = Doc(sys.argv[1])
heads = refs = 0
for i, b in enumerate(d.blocks):
    if not b.startswith("<w:p"):
        continue
    txt = d.text(i).strip()
    new = b
    m = re.match(rf"^({ROM})\.\s", txt)
    if m and txt.isupper():
        old = m.group(1)
        new = new.replace(f"<w:t xml:space=\"preserve\">{old}.  ",
                          f"<w:t xml:space=\"preserve\">{MAP[old]}.  ", 1)
        if new != b:
            heads += 1
    def fix(mm):
        global refs
        refs += 1
        return "Section " + MAP[mm.group(1)] + (mm.group(2) or "")
    new2 = re.sub(rf"Section ({ROM})(-[A-Z]\b)?", fix, new)
    d.blocks[i] = new2
d.save()
print(f"renumbered {heads} headings and {refs} cross-references")
