"""Build a white poster collection with Neff and three-way contact rankings.

Preprocess separately with prepare_poster_product.py. Reuse hash-checked white
artwork from export_poster.py for unchanged panels; replace all six standard
predictor contact panels from the new cached summaries. No model inference.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from PIL import Image

from prepare import sha256
from prepare_poster_product import CONTACT_FIGURES
from render import method_figure, static_depth
from theme import CAPTIONS, INK, NEFF_FIGURES, PAPER

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DESTINATION = 'hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/poster-neff-product-2026-10-13'
RANKING_NOTE = 'Structure contacts: pLDDT × pLDDT × contact degree'
CONTACT_CAPTION = ('Same 314 natural proteins (97 validation + 217 test), fixed 248B-token MarinFold. '
    'Standard structure predictors rank emitted ConFind degree>0 pairs by pLDDT_i × pLDDT_j × degree; '
    'residue pLDDT is mean heavy-atom confidence normalized to [0,1]. Filter resolved residues and separation '
    '≥6 (all) or ≥24 (long) before selection. Denominator is R true contacts, floor(L/5), or floor(L/2) '
    'as indicated; unfilled ranks receive zero credit. Truth is degree≥0.001 and does not enter ranking. '
    'MarinFold consensus and Sequence KNN scores are unchanged. Neff80 on frozen alignments; '
    'equal protein means and 95% bootstrap intervals. Source: data/poster_product_figure_rows.csv and '
    'data/poster_product_summary.csv; pair ordering in the published ranked_pairs.parquet. See POSTER_NEFF.md.')


def names_for(figure: str, metrics: list[str]) -> list[str]:
    """Match the renderer's stable metric filenames."""
    names = []
    for metric in metrics:
        suffix = '_lddt' if metric == 'lddt' else ('_long' if metric.endswith('_long') else '')
        if figure.removesuffix('_neff') == '04b_knn' and metric.startswith('r_precision'):
            suffix = '_r_precision' + ('_long' if metric.endswith('_long') else '')
        names.append(figure + suffix)
    return names


def chosen_names(all_names: set[str]) -> list[str]:
    """Keep every diagnostic, but only one depth/ranking version per metric."""
    excluded = {'01d_af3_depth_context'}
    for base, metrics in NEFF_FIGURES.items():
        excluded.update(names_for(base, metrics))
    excluded.update(name for name in all_names if name.startswith(('04c_', '04d_', '04e_')))
    selected = sorted(all_names - excluded)
    if len(selected) != 46:
        raise ValueError(f'Expected 46 curated plots, got {len(selected)}; review the inventory')
    return selected


def rasterize(task: tuple[Path, Path, int]) -> None:
    """Render a native vector at the requested width, with 300-dpi tagging."""
    pdf, png, width = task
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '300', '-scale-to-x', str(width),
                    '-scale-to-y', '-1', '-W', str(width), str(pdf), str(png.with_suffix(''))], check=True)
    with Image.open(png) as image:
        image.save(png, dpi=(300, 300))


def main() -> None:
    """Publish a clearly indexed PDF/PNG/SVG collection and numerical lineage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    analysis = json.loads((DATA/'poster_product_analysis.json').read_text())
    for filename, expected in analysis['files'].items():
        if sha256(DATA/filename) != expected:
            raise ValueError(f'Changed prepared table: {filename}')
    for filename, expected in analysis['inputs'].items():
        if sha256(DATA/filename) != expected:
            raise ValueError(f'Changed analysis input: {filename}; rerun prepare_poster_product.py')
    sampling = json.loads((DATA/'pl2_sampling_analysis.json').read_text())
    for filename, expected in sampling['files'].items():
        if sha256(DATA/filename) != expected:
            raise ValueError(f'Changed sampling table: {filename}; rerun prepare_sampling_pl2.py')
    comparison = json.loads((DATA/'pl2_sampling_comparison_analysis.json').read_text())
    for filename, expected in {**comparison['files'], **comparison['inputs']}.items():
        if sha256(DATA/filename) != expected:
            raise ValueError(f'Changed comparison table: {filename}; rerun prepare_sampling_comparison.py')
    base = HERE/'scratch/poster_public/poster_plots'
    old_manifest = json.loads((DATA/'poster_manifest.json').read_text())
    old = {row['name']: row for row in old_manifest['figures']}
    names = chosen_names(set(old))
    replacements = {name for figure, metrics in CONTACT_FIGURES.items() for name in names_for(figure, metrics)} | {'03_method'}
    stage = HERE/'scratch/poster_neff_product_public'
    bundle = stage/'poster_neff_product'
    if stage.exists():
        shutil.rmtree(stage)
    for folder in ['pdf', 'png', 'svg', 'analysis', 'fonts']:
        (bundle/folder).mkdir(parents=True)
    vectors = HERE/'scratch/poster_neff_product_vectors'
    vectors.mkdir(exist_ok=True)
    for font in (DATA/'inputs').glob('Lato-*.ttf'):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': 'Lato', 'font.size': 11, 'text.color': INK,
                         'axes.labelcolor': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': PAPER})
    summary = pd.read_csv(DATA/'poster_product_summary.csv')
    for figure, metrics in CONTACT_FIGURES.items():
        for metric in metrics:
            static_depth(summary, figure, metric, poster_dir=vectors,
                         ranking_note=RANKING_NOTE, caption=CONTACT_CAPTION)
    method_figure(pd.read_csv(DATA/'training_sources.csv'), poster_dir=vectors)
    previews = HERE/'plots/poster_neff_product'
    previews.mkdir(exist_ok=True)
    tasks = []
    for name in names:
        for extension in ['pdf', 'svg', 'png']:
            target = bundle/extension/f'{name}.{extension}'
            if name in replacements:
                if extension == 'png':
                    tasks.append((bundle/'pdf'/f'{name}.pdf', target, 7200))
                    continue
                source = vectors/f'{name}.{extension}'
            else:
                source = base/extension/f'{name}.{extension}'
                key = 'png_sha256' if extension == 'png' else f'poster_{extension}_sha256'
                if sha256(source) != old[name][key]:
                    raise ValueError(f'Stale base artwork: {source}; rerun export_poster.py')
            shutil.copyfile(source, target)
        if name in replacements:
            shutil.copyfile(bundle/'pdf'/f'{name}.pdf', previews/f'{name}.pdf')
            tasks.append((bundle/'pdf'/f'{name}.pdf', previews/f'{name}.png', 1800))
            caption = CONTACT_CAPTION if name != '03_method' else CAPTIONS['03_method'] + ' Poster schematic uses the main top-L/2 Helico arm.'
            (previews/f'{name}.png.meta.json').write_text(json.dumps(dict(script='export_poster_neff.py', args=[], caption=caption), indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(rasterize, tasks))
    # The six contact panels carry precisely the prepared rows they display.
    # Other panels retain hash-pinned provenance from the full poster bundle.
    for filename in [*analysis['files'], *sampling['files'], *comparison['files'],
                     'pl2_sampling_analysis.json', 'pl2_sampling_comparison_analysis.json',
                     'poster_product_analysis.json', 'neff_figure_rows.csv', 'neff_summary.csv',
                     'neff_af3_context_rows.csv', 'neff_af3_context_summary.csv', 'msa_neff.csv', 'af3_notice.txt', 'af3_output_terms.md']:
        shutil.copyfile(DATA/filename, bundle/'analysis'/filename)
    for source in (DATA/'inputs').glob('Lato-*'):
        shutil.copyfile(source, bundle/'fonts'/source.name)
    shutil.copyfile(HERE/'POSTER_NEFF.md', bundle/'README.md')
    shutil.copyfile(HERE/'SAMPLING_L2.md', bundle/'SAMPLING_L2.md')
    shutil.copyfile(DATA/'poster_manifest.json', bundle/'analysis/original_poster_manifest.json')
    records = []
    for name in names:
        with Image.open(bundle/'png'/f'{name}.png') as image:
            width, height = image.size
            if image.convert('RGB').getpixel((0, 0)) != (255, 255, 255):
                raise ValueError(f'Nonwhite poster canvas: {name}')
        contact = name in replacements and name != '03_method'
        source_summary = 'analysis/original_poster_manifest.json'
        if contact:
            source_summary = 'analysis/poster_product_summary.csv'
        elif name in {'06_sampling_pl2', '06_sampling_pl2_long'}:
            source_summary = 'analysis/pl2_sampling_comparison_summary.csv'
        records.append(dict(name=name, width_px=width, height_px=height, dpi=300,
            background='white', depth_axis='neff80' if '_neff' in name else 'not stratified',
            standard_contact_ranking='plddt_product_degree' if contact or name.startswith('06_sampling_pl2') else 'not applicable',
            source_summary=source_summary,
            caption=CONTACT_CAPTION if contact else old[name]['caption'],
            files={extension:dict(path=f'{extension}/{name}.{extension}', sha256=sha256(bundle/extension/f'{name}.{extension}'))
                   for extension in ['pdf', 'png', 'svg']}))
    manifest = dict(destination=DESTINATION, figure_count=len(records), background='white', png_width_px=7200,
                    figures=records, analysis_sha256=sha256(DATA/'poster_product_analysis.json'),
                    sampling_analysis_sha256=sha256(DATA/'pl2_sampling_analysis.json'),
                    sampling_comparison_sha256=sha256(DATA/'pl2_sampling_comparison_analysis.json'),
                    base_publication=old_manifest['destination'],
                    preprocess_command='uv run python prepare_poster_product.py',
                    sampling_preprocess_command='uv run python prepare_sampling_pl2.py',
                    comparison_preprocess_command='uv run python prepare_sampling_comparison.py',
                    render_command='uv run python export_poster_neff.py --upload')
    content = json.dumps(manifest, indent=2)+'\n'
    (bundle/'manifest.json').write_text(content)
    (DATA/'poster_neff_product_manifest.json').write_text(content)
    cards = ''.join(f'<article><h2>{html.escape(name)}</h2><a href="pdf/{name}.pdf"><img loading="lazy" src="png/{name}.png" alt="{html.escape(name)}"></a><p><a href="pdf/{name}.pdf">PDF</a> · <a href="png/{name}.png">PNG</a> · <a href="svg/{name}.svg">SVG</a></p></article>' for name in names)
    (bundle/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MarinFold poster plots</title>'
        '<style>body{font:16px sans-serif;background:white;color:#222;max-width:1200px;margin:40px auto}article{margin:40px 0;border-top:1px solid #ddd}img{width:100%;height:auto}a{color:#385c8f}</style>'
        f'<h1>MarinFold poster plots</h1><p>White backgrounds · Neff depth panels · three-way structural-contact ranking · {len(names)} figures</p>' + cards)
    subprocess.run(['pdfunite', *[str(bundle/'pdf'/f'{name}.pdf') for name in names], str(bundle/'all_plots.pdf')], check=True)
    with zipfile.ZipFile(stage/'poster_neff_product.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(stage))
    provenance = stage/'reproduce'
    provenance.mkdir()
    for filename in ['prepare_poster_product.py', 'prepare_sampling_pl2.py', 'prepare_sampling_comparison.py', 'prepare_pl5.py', 'prepare_neff.py',
                     'generation/score_contacts.py', 'export_poster_neff.py', 'render.py', 'prepare.py', 'poster_style.py', 'theme.py',
                     'build_summary.py', 'test_poster_product.py', 'render_af3_context.py', 'render_af3_sampling.py', 'render_contact_confidence.py',
                     'render_oracle_budget.py', 'render_seed_completion.py', 'pyproject.toml', 'uv.lock', 'POSTER_NEFF.md', 'SAMPLING_L2.md']:
        (provenance/filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/filename, provenance/filename)
    ranking_path = HERE/analysis['ranked_pairs']['path']
    if sha256(ranking_path) != analysis['ranked_pairs']['sha256']:
        raise ValueError('Changed ranked pair archive')
    shutil.copyfile(ranking_path, provenance/'ranked_pairs.parquet')
    publication = dict(destination=DESTINATION, files={str(path.relative_to(stage)):dict(bytes=path.stat().st_size, sha256=sha256(path))
        for path in sorted(stage.rglob('*')) if path.is_file()})
    content = json.dumps(publication, indent=2)+'\n'
    (stage/'publication_manifest.json').write_text(content)
    (DATA/'poster_neff_product_publication.json').write_text(content)
    print(f'{len(records)} figures; PDF + PNG + SVG; ZIP {(stage/"poster_neff_product.zip").stat().st_size/1e6:.1f} MB', flush=True)
    if args.upload:
        subprocess.run(['hf', 'buckets', 'sync', str(stage), DESTINATION, '--quiet'], check=True)


if __name__ == '__main__':
    main()
