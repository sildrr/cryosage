import pytest

from webapp.core import SHOWCASE


@pytest.mark.parametrize("dataset,workspace,job,resolution", [
    ("10208", "W12", "J282", 3.793032172689106),
    ("10012", "W13", "J306", 3.1900709437221684),
])
def test_verified_p18_showcase(dataset, workspace, job, resolution):
    entry = next(row for row in SHOWCASE if row["id"] == dataset)
    assert entry["project"] == "P18"
    assert entry["workspace"] == workspace
    assert entry["job"] == job
    assert entry["resolution"] == pytest.approx(resolution)
    assert "fsc_info.radwn_final_A" in entry["source"]


def test_showcase_dataset_ids_unique():
    ids = [row["id"] for row in SHOWCASE]
    assert len(ids) == len(set(ids))


def test_showcase_images_and_provenance():
    import json
    from webapp.core import REPO
    directory = REPO / "webapp/static/showcase"
    sources = {row["dataset_id"]: row for row in json.loads((directory / "sources.json").read_text())}
    for case in SHOWCASE:
        for kind in ("shape", "fsc"):
            assert (directory / f"{case['id']}-{kind}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert sources[case["id"]]["job"] == case["job"]
        assert sources[case["id"]]["fsc_job"] == ("J118" if case["id"] == "11280" else case["job"])


def test_showcase_uses_uniform_final_plots():
    from webapp.core import REPO
    js = (REPO / "webapp/static/app.js").read_text()
    section = js[js.index("async function showcase()"):js.index("$('example').onclick")]
    assert "mini-chart" not in section
    assert "createElementNS" not in section
    assert "['shape','3D Density Map · Single View']" in section
    assert "['fsc','Final FSC Curve']" in section
    assert section.index("['fsc','Final FSC Curve']") < section.index("['shape','3D Density Map · Single View']")


def test_showcase_full_width_rows_and_responsive_panels():
    from webapp.core import REPO
    css = (REPO / "webapp/static/style.css").read_text()
    assert ".results-section .result-grid{grid-template-columns:minmax(0,1fr)}" in css
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in css
    assert "@media(max-width:760px){.result-card .showcase-previews{grid-template-columns:minmax(0,1fr)" in css


def test_showcase_curves_are_numeric_samples_with_frequency_axes():
    import json
    import math
    from webapp.core import REPO
    directory = REPO / "webapp/static/showcase"
    for case in SHOWCASE:
        curves = json.loads((directory / f"{case['id']}-curves.json").read_text())
        assert len(curves) == 3
        for curve in curves:
            frequency, values = curve["frequency_inv_A"], curve["values"]
            assert len(frequency) == len(values) > 20
            assert all(math.isfinite(v) for v in frequency + values)
            assert all(a < b for a, b in zip(frequency, frequency[1:]))
        assert curves[0]["field"] == ("J118/J118_fsc_mask.xml" if case["id"] == "11280" else "fsc_autotight_noisesub")
