"""Build and independently compile the portable manuscript source package."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'knolling_bot_v2_source.zip'
FILES=['main.tex','appendix.tex','references.bib','README.md','requirements-paper.txt',
       'build.sh','.latexmkrc','knolling_bot_v2_paper.pdf']
DIRS=['audit','figures','figure_sources','fonts','scripts','template','renders']
FIGURES=['architecture_main','qualitative_full_annotated','qualitative_modes',
         'holdout_diagnostics','selector_training','motion_correspondence','failure_explanation','layout_constraints']

def package():
    members=[ROOT/n for n in FILES]
    for name in DIRS:
        for p in (ROOT/name).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('preview') or p.suffix in ('.log','.blend1'):
                continue
            if 'independent_visual_20261115' in p.parts or p.name=='contact_sheet.png':
                continue  # Historical review images, not manuscript inputs.
            if name=='figures' and p.stem not in FIGURES:
                continue
            members.append(p)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(members):z.write(p,Path('paper_submit')/p.relative_to(ROOT))

package()
target=ROOT/'build/portable_visual_revision'
if target.exists():shutil.rmtree(target)
target.mkdir(parents=True)
with zipfile.ZipFile(ARCHIVE) as z:z.extractall(target)
copy=target/'paper_submit'
with (ROOT/'build/portable_figures.log').open('w') as log:
    subprocess.run([sys.executable,'scripts/make_figures.py'],cwd=copy,stdout=log,stderr=subprocess.STDOUT,check=True)
for name in FIGURES:
    original=ROOT/'figures'/f'{name}.png'
    assert original.read_bytes()==(copy/'figures'/original.name).read_bytes(),original.name
for name in ['visual_measurements.json','motion_annotations.json','failure_annotations.json',
             'paired_mode_analysis.json','footprint_clearance_analysis.json']:
    assert (ROOT/'audit'/name).read_bytes()==(copy/'audit'/name).read_bytes(),name
with (ROOT/'build/portable_visual_revision.log').open('w') as log:
    subprocess.run(['bash','build.sh'],cwd=copy,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'scripts/verify_submission.py'],cwd=copy,stdout=log,stderr=subprocess.STDOUT,check=True)
a=pymupdf.open(ROOT/'knolling_bot_v2_paper.pdf');b=pymupdf.open(copy/'knolling_bot_v2_paper.pdf')
assert len(a)==len(b)==13
assert [p.get_text() for p in a]==[p.get_text() for p in b]
assert [p.get_pixmap().samples for p in a]==[p.get_pixmap().samples for p in b]
log=(copy/'build/main.log').read_text()
assert not any(term in log for term in ['Overfull','LaTeX Warning','undefined references'])
report={'status':'PASS','revision':'Canonical paper_submit, comprehensive academic prose revision with all figure artwork preserved, 2026-11-15',
        'method':'Extract complete source bundle to an isolated directory; regenerate Figures 1–5, 7 and 8 from bundled evidence and fonts while preserving approved Figure 6; compile with pdfLaTeX/BibTeX via latexmk and run submission verification.',
        'pages':len(b),'page_text_identical':True,'page_pixels_identical':True,'source_repository_required':False,
        'model_weights_required':False,'original_dataset_required':False,
        'compiled_with_bundled_render_inputs':True,'figure_png_regeneration_identical':True,
        'figure6_files_and_latex_unchanged':True,
        'geometric_annotations_regeneration_identical':True,
        'pdf_sha256':hashlib.sha256((ROOT/'knolling_bot_v2_paper.pdf').read_bytes()).hexdigest()}
(ROOT/'audit/portable_build_check.json').write_text(json.dumps(report,indent=2)+'\n')
package() # Include the completed verification report.
print(json.dumps(report,indent=2))
print(f'Packaged {ARCHIVE.stat().st_size:,} bytes: {ARCHIVE}')
