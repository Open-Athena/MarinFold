"""Browser checks: all 97 visual selections equal the canonical scores."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Exercise the offline report at desktop and mobile sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Existing Chromium executable, if installed outside Playwright")
    args = parser.parse_args()
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=args.browser, args=["--no-sandbox"])
        for name, width, height in (("desktop", 1440, 1100), ("mobile", 390, 844)):
            errors = []
            page = browser.new_page(viewport={"width": width, "height": height})
            page.on("pageerror", lambda error, collected=errors: collected.append(str(error)))
            page.goto((HERE / "report.html").as_uri())
            page.wait_for_function("document.documentElement.dataset.ready === 'true'")
            assert page.locator(".card").count() == 97
            assert page.locator("#protein-select option").count() == 97
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            discrepancies = page.evaluate("""() => DATA.proteins.filter(p => {
                const tp = p.topR.filter(([i,j]) => p.truthSet.has(key(i,j))).length;
                return Math.abs(tp / p.topR.length - p.metrics.r_precision) > 1e-12;
            }).map(p => p.stem)""")
            assert not discrepancies, discrepancies
            page.select_option("#view-mode", "errors")
            page.select_option("#range", "long")
            page.select_option("#budget", "L5")
            page.select_option("#protein-select", "7y8h_A")
            assert "84.3%" in page.locator("#protein-stats").inner_text()
            page.select_option("#budget", "threshold")
            assert page.locator("#vote-control").is_visible()
            page.locator("#vote-threshold").fill("100")
            page.locator("#vote-threshold").dispatch_event("input")
            page.fill("#search", "8arl")
            assert page.locator(".card").count() == 1
            page.locator(".card").click()
            assert page.locator("#protein-title").inner_text() == "8arl_A"
            page.fill("#search", "")
            page.select_option("#sort", "best")
            assert page.locator(".card").first.get_attribute("data-stem") == "7y8h_A"
            page.evaluate("document.querySelectorAll('.card').forEach(paintCard)")
            assert page.locator(".card[data-painted=yes]").count() == 97
            page.select_option("#budget", "R")
            page.select_option("#range", "all")
            page.locator("#pred-map").scroll_into_view_if_needed()
            box = page.locator("#pred-map").bounding_box()
            assert box is not None
            page.mouse.move(box["x"] + box["width"] * .25, box["y"] + box["height"] * .25)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * .65, box["y"] + box["height"] * .65)
            page.mouse.up()
            assert page.evaluate("zoom !== null && zoom.x1 - zoom.x0 < current.L")
            page.click("#reset-zoom")
            assert page.evaluate("zoom === null")
            with page.expect_download() as download:
                page.click("#download-pairs")
            csv_path = Path(download.value.path())
            assert len(csv_path.read_text().splitlines()) == 261
            assert not errors, errors
            print(f"{name}: 97 exact score matches; rendering, controls, download and layout passed")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
