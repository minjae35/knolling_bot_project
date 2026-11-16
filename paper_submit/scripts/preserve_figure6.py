"""Ensure the user's approved Figure 6 remains exactly unchanged."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify():
    record=json.loads((ROOT/'audit/figure6_frozen.json').read_text())
    for path,expected in record['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    blocks=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',(ROOT/'appendix.tex').read_text(),re.S)
    actual=next(x for x in blocks if 'qualitative_modes.pdf' in x)
    assert actual==record['latex_figure_block'],'Figure 6 LaTeX block changed'
    return {'status':'PASS','unchanged_files':len(record['sha256']),'latex_block_unchanged':True}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
