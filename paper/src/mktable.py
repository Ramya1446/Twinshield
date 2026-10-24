"""Build an IEEE-Access-style <w:tbl> matching the ones already in the paper."""
from xml.sax.saxutils import escape

TBL_PR = ('<w:tblPr><w:tblStyle w:val="a1"/><w:tblW w:w="{tw}" w:type="dxa"/>'
          '<w:jc w:val="center"/><w:tblInd w:w="0" w:type="dxa"/><w:tblBorders>'
          '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
          '<w:left w:val="nil"/>'
          '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
          '<w:right w:val="nil"/>'
          '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
          '<w:insideV w:val="nil"/></w:tblBorders>'
          '<w:tblLayout w:type="fixed"/>'
          '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" '
          'w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>')

PPR = ('<w:pPr><w:spacing w:before="20" w:after="20" w:line="200" w:lineRule="auto"/>'
       '{jc}<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr></w:pPr>')


def cell(text, w, bold=False, centre=True):
    jc = '<w:jc w:val="center"/>' if centre else ''
    rpr = ('<w:rPr>' + ('<w:b/><w:bCs/>' if bold else '') +
           '<w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr>')
    runs = ''
    for k, line in enumerate(str(text).split('\n')):
        if k:
            runs += '<w:r>' + rpr + '<w:br/></w:r>'
        runs += ('<w:r>' + rpr + '<w:t xml:space="preserve">' + escape(line) +
                 '</w:t></w:r>')
    return ('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:vAlign w:val="center"/>'
            '</w:tcPr><w:p>' % w + PPR.format(jc=jc) + runs + '</w:p></w:tc>')


def row(cells, widths, bold=False, header=False, first_left=True):
    tr_pr = ('<w:trPr>' + ('<w:tblHeader/>' if header else '') +
             '<w:jc w:val="center"/></w:trPr>')
    out = ''
    for k, (c, w) in enumerate(zip(cells, widths)):
        centre = not (k == 0 and first_left and not header)
        out += cell(c, w, bold=bold or header, centre=centre)
    return '<w:tr>' + tr_pr + out + '</w:tr>'


def table(header, rows, widths, bold_rows=()):
    tw = sum(widths)
    grid = '<w:tblGrid>' + ''.join('<w:gridCol w:w="%d"/>' % w for w in widths) + '</w:tblGrid>'
    body = row(header, widths, header=True)
    for i, r in enumerate(rows):
        body += row(r, widths, bold=(i in bold_rows))
    return '<w:tbl>' + TBL_PR.format(tw=tw) + grid + body + '</w:tbl>'
