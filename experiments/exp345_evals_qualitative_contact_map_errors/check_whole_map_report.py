"""Check all embedded map scores and exercise the full offline rollout viewer."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Cross-check 19,400 rendered maps against Python scores on desktop/mobile."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser")
    args = parser.parse_args()
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=args.browser, args=["--no-sandbox"])
        for name, width, height in (("desktop", 1440, 1100), ("mobile", 390, 844)):
            errors = []
            page = browser.new_page(viewport={"width": width, "height": height})
            page.on("pageerror", lambda error, collected=errors: collected.append(str(error)))
            page.goto((HERE / "whole_map_report.html").as_uri())
            page.wait_for_function("document.documentElement.dataset.ready === 'true'", timeout=120000)
            assert page.locator(".card").count() == 97
            assert page.locator("#protein-select option").count() == 97
            assert page.locator("#sample-select option").count() == 200
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            check = page.evaluate("""() => {
                let count=0;const bad=[];
                for(const p of DATA.proteins)for(let n=0;n<200;n++){
                    for(const range of ['all','long']){
                        const actual=metrics(p,sample(p,n),range).f1;
                        const expected=(range==='all'?p.f1:p.long_f1)[n];
                        if(Math.abs(actual-expected)>5.1e-9)bad.push([p.stem,n,range]);
                    }count++;
                }return{count,bad};
            }""")
            assert check == {"count": 19400, "bad": []}, check
            assert page.evaluate("""() => {
                const mean = DATA.proteins.reduce((total,p) => total + metrics(p,p.pooled).f1,0)/97;
                return Math.abs(mean-statistic('pooled_200').mean) < 1e-12 &&
                    DATA.proteins.every(p => p.f1[p.selection.all.oracle_200] === Math.max(...p.f1));
            }""")
            page.select_option("#variant", "oracle")
            assert page.locator("#oracle-note").is_visible()
            assert page.evaluate("selectedIndex === current.selection.all.oracle_200")
            page.select_option("#variant", "consensus")
            assert not page.locator("#oracle-note").is_visible()
            assert page.evaluate("selectedIndex === current.selection.all.consensus_A")
            page.select_option("#variant", "likelihood")
            assert page.evaluate("selectedIndex === current.selection.all.likelihood_A")
            page.select_option("#sample-select", "199")
            assert page.evaluate("selectedIndex === 199 && $('variant').value === 'sample'")
            page.select_option("#view-mode", "errors")
            page.select_option("#range", "long")
            page.select_option("#protein-select", "7y8h_A")
            assert page.locator("#protein-title").inner_text() == "7y8h_A"
            page.fill("#search", "8arl")
            assert page.locator(".card").count() == 1
            page.locator(".card").click()
            assert page.locator("#protein-title").inner_text() == "8arl_A"
            page.fill("#search", "")
            page.select_option("#sort", "name")
            assert page.locator(".card").first.get_attribute("data-stem") == page.evaluate("DATA.proteins.map(p=>p.stem).sort()[0]")
            page.evaluate("document.querySelectorAll('.card').forEach(paintCard)")
            assert page.locator(".card[data-painted=yes]").count() == 97
            page.select_option("#range", "all")
            page.locator("#pred-map").scroll_into_view_if_needed()
            box = page.locator("#pred-map").bounding_box()
            assert box is not None
            page.mouse.move(box["x"]+box["width"]*.25, box["y"]+box["height"]*.25)
            page.mouse.down()
            page.mouse.move(box["x"]+box["width"]*.65, box["y"]+box["height"]*.65)
            page.mouse.up()
            assert page.evaluate("zoom !== null && zoom.x1-zoom.x0 < current.L")
            page.click("#reset-zoom")
            assert page.evaluate("zoom === null")
            expected_rows = page.evaluate("selected.length")
            with page.expect_download() as download:
                page.click("#download-pairs")
            assert len(Path(download.value.path()).read_text().splitlines()) == expected_rows+1
            assert not errors, errors
            page.screenshot(path=str(HERE / f".cache/whole-map-{name}.png"), full_page=False)
            print(f"{name}: 19,400 maps × 2 ranges match; 291 canvases and controls pass")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
