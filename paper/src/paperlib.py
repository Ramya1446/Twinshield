# -*- coding: utf-8 -*-
"""Rebuild the IEEE Access paper's body, reusing the template's own styles."""
import re, sys
sys.path.insert(0, '/tmp/claude-0/-home-user-Twinshield/28c50788-17b6-5fcf-b899-a73664e0ef5c/scratchpad/sc_scripts')
from docutil import Doc
from xml.sax.saxutils import escape


def shell(tpl, runs):
    """Clone a paragraph's formatting, replacing its runs. `runs` is text or
    a list of (text, bold) pairs."""
    rs = list(re.finditer(r'<w:r\b(?:(?!</w:r>).)*?</w:r>', tpl, re.S))
    assert rs, "paragraph has no run to clone"
    base = rs[0].group(0)
    rpr = re.search(r'<w:rPr\b.*?</w:rPr>', base, re.S)
    rpr = rpr.group(0) if rpr else ''
    # the cloned run may itself be bold, so build both variants explicitly
    rpr_plain = rpr.replace('<w:b/>', '').replace('<w:bCs/>', '')
    rpr_b = rpr_plain
    if rpr_b.startswith('<w:rPr>'):
        rpr_b = rpr_b.replace('<w:rPr>', '<w:rPr><w:b/><w:bCs/>', 1)
    else:
        rpr_b = '<w:rPr><w:b/><w:bCs/></w:rPr>'
    rpr = rpr_plain
    if isinstance(runs, str):
        runs = [(runs, False)]
    body = ''
    for text, bold in runs:
        for k, line in enumerate(text.split('\n')):
            if k:
                body += '<w:r>' + (rpr_b if bold else rpr) + '<w:br/></w:r>'
            body += ('<w:r>' + (rpr_b if bold else rpr) +
                     '<w:t xml:space="preserve">' + escape(line) + '</w:t></w:r>')
    return tpl[:rs[0].start()] + body + tpl[rs[-1].end():]


class Paper:
    def __init__(self, path):
        self.d = Doc(path)
        self.t = {}

    def grab(self, key, frag):
        hits = [i for i, b in enumerate(self.d.blocks)
                if b.startswith('<w:p') and frag in self.d.text(i)]
        assert hits, f"template {key}: {frag!r} not found"
        self.t[key] = self.d.blocks[hits[0]]
        return self.t[key]

    def p(self, key, runs):
        return shell(self.t[key], runs)
