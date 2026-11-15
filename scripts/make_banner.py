"""Generates the README title banner (light and dark variants)."""
from pathlib import Path
L = {
 'O':[" ██████╗ ","██╔═══██╗","██║   ██║","██║   ██║","╚██████╔╝"," ╚═════╝ "],
 'P':["██████╗ ","██╔══██╗","██████╔╝","██╔═══╝ ","██║     ","╚═╝     "],
 'S':["███████╗","██╔════╝","███████╗","╚════██║","███████║","╚══════╝"],
 'G':[" ██████╗ ","██╔════╝ ","██║  ███╗","██║   ██║","╚██████╔╝"," ╚═════╝ "],
 'Y':["██╗   ██╗","╚██╗ ██╔╝"," ╚████╔╝ ","  ╚██╔╝  ","   ██║   ","   ╚═╝   "],
 'M':["███╗   ███╗","████╗ ████║","██╔████╔██║","██║╚██╔╝██║","██║ ╚═╝ ██║","╚═╝     ╚═╝"],
}
ROWS = ["".join(L[c][i] for c in "OPSGYM") for i in range(6)]
W = max(len(r) for r in ROWS); CW, CH, PAD = 11, 20, 24
SEGS = {'═':[(0,.5,1,.5)], '║':[(.5,0,.5,1)], '╗':[(0,.5,.5,.5),(.5,.5,.5,1)],
        '╔':[(.5,.5,1,.5),(.5,.5,.5,1)], '╝':[(0,.5,.5,.5),(.5,0,.5,.5)],
        '╚':[(.5,.5,1,.5),(.5,0,.5,.5)]}

def svg(ink, shadow, sub):
    w, h = W*CW + 2*PAD, 6*CH + 2*PAD + 44
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" role="img" aria-label="OpsGym">']
    for y, row in enumerate(ROWS):
        for x, c in enumerate(row):
            X, Y = PAD + x*CW, PAD + y*CH
            if c == '█':
                out.append(f'<rect x="{X}" y="{Y}" width="{CW+0.4}" height="{CH+0.4}" fill="{ink}"/>')
            elif c in SEGS:
                for a, b, c2, d in SEGS[c]:
                    out.append(f'<line x1="{X+a*CW}" y1="{Y+b*CH}" x2="{X+c2*CW}" y2="{Y+d*CH}" '
                               f'stroke="{shadow}" stroke-width="2.4" stroke-linecap="square"/>')
    out.append(f'<text x="{w/2}" y="{PAD+6*CH+32}" text-anchor="middle" '
               'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
               f'font-size="15" fill="{sub}">does a loose verifier teach an agent to cut corners?</text>')
    out.append('</svg>')
    return "\n".join(out)

out = Path("docs/assets"); out.mkdir(parents=True, exist_ok=True)
(out / "banner-light.svg").write_text(svg('#1f2937', '#9ca3af', '#4b5563'), encoding="utf-8")
(out / "banner-dark.svg").write_text(svg('#e6edf3', '#6e7681', '#8b949e'), encoding="utf-8")
