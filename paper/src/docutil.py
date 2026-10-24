"""Minimal top-level block editor for word/document.xml."""
import re

BLOCK = re.compile(r'<w:p\b[^>]*/>|<w:p\b[^>]*>.*?</w:p>|<w:tbl>.*?</w:tbl>|'
                   r'<w:sectPr\b[^>]*>.*?</w:sectPr>|<w:sectPr\b[^>]*/>', re.S)


class Doc:
    def __init__(self, path):
        self.path = path
        self.xml = open(path, encoding='utf-8').read()
        a = self.xml.index('<w:body>') + len('<w:body>')
        b = self.xml.rindex('</w:body>')
        self.head, self.tail = self.xml[:a], self.xml[b:]
        body = self.xml[a:b]
        self.blocks, last = [], 0
        for m in BLOCK.finditer(body):
            if m.start() > last:
                gap = body[last:m.start()]
                if gap.strip():
                    self.blocks.append(gap)
            self.blocks.append(m.group(0))
            last = m.end()
        if body[last:].strip():
            self.blocks.append(body[last:])

    def pidx(self):
        """Map 1-based paragraph number -> block index."""
        out, n = {}, 0
        for i, b in enumerate(self.blocks):
            if b.startswith('<w:p'):
                n += 1
                out[n] = i
        return out

    def text(self, i):
        return ''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', self.blocks[i], re.S))

    def save(self, path=None):
        out = self.head + ''.join(self.blocks) + self.tail
        open(path or self.path, 'w', encoding='utf-8').write(out)
