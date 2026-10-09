#!/usr/bin/env python3
"""Reproduce the external-incidence-denominator sensitivity analysis.

This repository contains a deterministic, figure-based approximation. It does
not reproduce the target article's Bayesian model or recover its posterior
draws. The script accepts all paths as command-line arguments so it can be
run from an anonymised repository on any operating system.

Example
-------
python run_analysis.py \
    --gbd data/raw/gbd_stomach_incidence_2022_both.csv \
    --globocan data/raw/globocan_stomach_age_groups_2022.csv \
    --figure-pdf data/raw/target_article.pdf \
    --output results
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pymupdf as fitz
from PIL import Image


GBD_AGE_RE = re.compile(r"^(\d+)-(\d+) years$")
GCO_AGE_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbd", type=Path, required=True, help="GBD age-specific incidence CSV")
    parser.add_argument("--globocan", type=Path, required=True, help="GLOBOCAN/NCC-aligned age-group CSV")
    parser.add_argument("--figure-pdf", type=Path, required=True, help="PDF containing the published Figure 4b")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--page", type=int, default=9, help="1-based PDF page containing Figure 4b (default: 9)")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def parse_gbd_age(age_name: str) -> float | None:
    if age_name == "95+ years":
        return 97.5
    match = GBD_AGE_RE.match(str(age_name))
    if not match:
        return None
    return (int(match.group(1)) + int(match.group(2))) / 2


def load_gbd(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "location_name", "year", "sex_name", "measure_name", "metric_name",
        "cause_name", "age_name", "val",
    ]
    require_columns(frame, required, "GBD input")
    mask = (
        frame["location_name"].eq("China")
        & frame["year"].astype(str).eq("2022")
        & frame["sex_name"].eq("Both")
        & frame["measure_name"].eq("Incidence")
        & frame["metric_name"].eq("Number")
        & frame["cause_name"].str.casefold().eq("stomach cancer")
    )
    frame = frame.loc[mask].copy()
    frame["age_midpoint"] = frame["age_name"].map(parse_gbd_age)
    frame = frame.loc[frame["age_midpoint"].notna() & frame["age_midpoint"].ge(20)].copy()
    frame = frame.sort_values("age_midpoint")
    if frame["age_name"].duplicated().any():
        raise ValueError("GBD input contains duplicate age groups after filtering")
    expected_starts = list(range(20, 95, 5)) + [95]
    actual_starts = [int(GBD_AGE_RE.match(x).group(1)) if GBD_AGE_RE.match(x) else 95 for x in frame["age_name"]]
    if actual_starts != expected_starts:
        raise ValueError(
            "GBD input must contain non-overlapping age groups 20–24 through 95+. "
            f"Found: {frame['age_name'].tolist()}"
        )
    return frame.reset_index(drop=True)


def load_globocan(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, ["age_group", "cases"], "GLOBOCAN input")
    frame["age_group"] = frame["age_group"].astype(str)
    frame["cases"] = pd.to_numeric(frame["cases"], errors="raise")
    required_groups = [f"{start}-{start + 4}" for start in range(20, 75, 5)]
    missing = [x for x in required_groups if x not in set(frame["age_group"])]
    if missing:
        raise ValueError(f"GLOBOCAN input is missing age groups: {missing}")
    if "75+" not in set(frame["age_group"]):
        all_age = frame.loc[frame["age_group"].eq("All ages"), "cases"]
        under_20 = frame.loc[frame["age_group"].isin(["0-4", "5-9", "10-14", "15-19"]), "cases"]
        if len(all_age) != 1 or len(under_20) != 4:
            raise ValueError("GLOBOCAN input needs either a 75+ row or all-age and four under-20 rows")
        residual = float(all_age.iloc[0] - under_20.sum())
        frame = pd.concat([frame, pd.DataFrame([{"age_group": "75+", "cases": residual}])], ignore_index=True)
    groups = required_groups + ["75+"]
    result = frame.set_index("age_group").loc[groups].copy()
    result["age_midpoint"] = [start + 2 for start in range(20, 75, 5)] + [77.5]
    return result.reset_index()


def extract_panel_images(page: fitz.Page) -> tuple[np.ndarray, np.ndarray, fitz.Rect, fitz.Rect]:
    """Find the right-hand heatmap and its right-hand colour bar by dimensions."""
    heat_candidates = []
    bar_candidates = []
    for image in page.get_images(full=True):
        xref = image[0]
        info = page.parent.extract_image(xref)
        width, height = info["width"], info["height"]
        rects = page.get_image_rects(xref)
        if width > 500 and height > 500:
            for rect in rects:
                heat_candidates.append((rect.x0, xref, rect))
        elif width < 100 and height > 400:
            for rect in rects:
                bar_candidates.append((rect.x0, xref, rect))
    if not heat_candidates or not bar_candidates:
        raise RuntimeError("Could not identify Figure 4b heatmap and colour bar images")
    _, heat_xref, heat_rect = max(heat_candidates, key=lambda item: item[0])
    _, bar_xref, bar_rect = max(bar_candidates, key=lambda item: item[0])
    heat_info = page.parent.extract_image(heat_xref)
    bar_info = page.parent.extract_image(bar_xref)
    heat = np.asarray(Image.open(io.BytesIO(heat_info["image"])).convert("RGB"), dtype=float)
    bar = np.asarray(Image.open(io.BytesIO(bar_info["image"])).convert("RGB"), dtype=float)
    return heat, bar, heat_rect, bar_rect


def locate_ticks(page: fitz.Page) -> tuple[list[float], list[float], list[float], list[float]]:
    """Locate the labelled axes and colour-bar ticks in the published vector PDF."""
    drawings = page.get_drawings()
    bar_ticks = sorted(
        [g["rect"].y0 for g in drawings if 448 < g["rect"].x0 < 449 and 449 < g["rect"].x1 < 450
         and g["rect"].height < 0.01 and 170 < g["rect"].y0 < 295], reverse=True
    )
    x_ticks = sorted(
        [g["rect"].x0 for g in drawings if 310.1 < g["rect"].y0 < 310.3 and 310.6 < g["rect"].y1 < 310.9
         and g["rect"].width < 0.01 and 275 < g["rect"].x0 < 425]
    )
    y_ticks = sorted(
        [g["rect"].y0 for g in drawings if 274.1 < g["rect"].x0 < 274.2 and 274.6 < g["rect"].x1 < 274.8
         and g["rect"].height < 0.01 and 150 < g["rect"].y0 < 312], reverse=True
    )
    if len(bar_ticks) != 6 or len(x_ticks) != 7 or len(y_ticks) != 5:
        raise RuntimeError(f"Unexpected vector tick count: colour bar={len(bar_ticks)}, x={len(x_ticks)}, y={len(y_ticks)}")
    return bar_ticks, x_ticks, y_ticks, drawings


class FigureSurface:
    def __init__(self, pdf_path: Path, page_number: int) -> None:
        self.pdf_path = pdf_path
        self.document = fitz.open(pdf_path)
        self.page = self.document[page_number - 1]
        self.heat, self.bar, self.heat_rect, self.bar_rect = extract_panel_images(self.page)
        bar_ticks, x_ticks, y_ticks, self.drawings = locate_ticks(self.page)
        self.bar_ticks = bar_ticks
        self.bar_values = np.arange(6) / 10
        self.bar_coef = np.polyfit(bar_ticks, self.bar_values, 1)
        bar_rows_y = self.bar_rect.y0 + (np.arange(self.bar.shape[0]) + 0.5) * self.bar_rect.height / self.bar.shape[0]
        self.bar_values_by_row = np.polyval(self.bar_coef, bar_rows_y)
        self.x_coef = np.polyfit(np.arange(1990, 2021, 5), x_ticks, 1)
        self.y_coef = np.polyfit(np.arange(0, 81, 20), y_ticks, 1)

    def decode(self, rgb: np.ndarray) -> float:
        palette = self.bar[:, self.bar.shape[1] // 2 - 2:self.bar.shape[1] // 2 + 3].mean(axis=1)
        distances = ((palette - rgb) ** 2).sum(axis=1)
        tied = np.flatnonzero(np.isclose(distances, distances.min(), atol=1e-10, rtol=0))
        return float(np.mean(self.bar_values_by_row[tied]))

    def sample(self, x_pdf: float, y_pdf: float, method: str = "median_patch") -> float:
        x = (x_pdf - self.heat_rect.x0) * self.heat.shape[1] / self.heat_rect.width - 0.5
        y = (y_pdf - self.heat_rect.y0) * self.heat.shape[0] / self.heat_rect.height - 0.5
        if not (-0.5 <= x <= self.heat.shape[1] - 0.5 and -0.5 <= y <= self.heat.shape[0] - 0.5):
            raise ValueError(f"Coordinate outside heatmap: {x}, {y}")
        x = float(np.clip(x, 0, self.heat.shape[1] - 1))
        y = float(np.clip(y, 0, self.heat.shape[0] - 1))
        if method == "bilinear":
            xi, yi = int(np.floor(x)), int(np.floor(y))
            xj, yj = min(xi + 1, self.heat.shape[1] - 1), min(yi + 1, self.heat.shape[0] - 1)
            return float(sum(self.decode(self.heat[b, a]) * w for a, b, w in [
                (xi, yi, (1 - x + xi) * (1 - y + yi)), (xj, yi, (x - xi) * (1 - y + yi)),
                (xi, yj, (1 - x + xi) * (y - yi)), (xj, yj, (x - xi) * (y - yi))]))
        xi, yi = int(round(x)), int(round(y))
        if method == "nearest":
            return self.decode(self.heat[yi, xi])
        patch = self.heat[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
        return float(np.median([self.decode(c) for c in patch.reshape(-1, 3)]))

    def prevalence(self, age: float, year: int = 2012, method: str = "median_patch") -> float:
        return self.sample(float(np.polyval(self.x_coef, year)), float(np.polyval(self.y_coef, age)), method)

    def contour_check(self) -> pd.DataFrame:
        contours = [g for g in self.drawings if g["color"] and all(c > 0.95 for c in g["color"])
                    and 274 <= g["rect"].x0 < 278 and g["rect"].x1 < 436
                    and g["rect"].width > 50 and 134 < g["rect"].y0 < 310]
        contours.sort(key=lambda g: g["rect"].y1, reverse=True)
        if len(contours) != 5:
            raise RuntimeError(f"Expected five labelled contours, found {len(contours)}")
        rows = []
        for contour_index, value in [(0, 0.1), (2, 0.3), (4, 0.5)]:
            points = sorted(set((float(p.x), float(p.y)) for item in contours[contour_index]["items"]
                                if item[0] == "l" for p in item[1:]))
            for x, y in points:
                if self.heat_rect.x0 + 2 < x < self.heat_rect.x1 - 2 and self.heat_rect.y0 + 2 < y < self.heat_rect.y1 - 2:
                    extracted = self.sample(x, y, "bilinear")
                    rows.append({"labelled_prevalence": value, "pdf_x": x, "pdf_y": y,
                                 "extracted_prevalence": extracted,
                                 "error_percentage_points": (extracted - value) * 100})
        return pd.DataFrame(rows)

    def audit(self, contour_df: pd.DataFrame) -> dict:
        palette = self.bar[:, self.bar.shape[1] // 2 - 2:self.bar.shape[1] // 2 + 3].mean(axis=1)
        _, counts = np.unique(palette, axis=0, return_counts=True)
        unique_rows = np.unique(palette, axis=0, return_index=True)[1]
        centres = sorted(np.mean(np.flatnonzero(np.all(palette == palette[i], axis=1))) for i in unique_rows)
        step = abs(self.bar_coef[0] * self.bar_rect.height / self.bar.shape[0])
        return {
            "pdf_sha256": hashlib.sha256(self.pdf_path.read_bytes()).hexdigest(),
            "pdf_page_1_based": self.document.page_count and self.page.number + 1,
            "heatmap_width_height": [int(self.heat.shape[1]), int(self.heat.shape[0])],
            "colourbar_height": int(self.bar.shape[0]),
            "colourbar_labelled_values": self.bar_values.tolist(),
            "nominal_prevalence_percentage_points_per_row": float(step * 100),
            "unique_palette_colors": int(len(unique_rows)),
            "max_rows_with_identical_colour": int(counts.max()),
            "contour_internal_check_n": int(len(contour_df)),
            "contour_mean_absolute_error_percentage_points": float(contour_df.error_percentage_points.abs().mean()),
            "contour_max_absolute_error_percentage_points": float(contour_df.error_percentage_points.abs().max()),
            "limits": "Figure resolution and contour checks are internal to the published graphic. No external model-value validation, probability error interval, or joint posterior reanalysis is performed.",
        }


def paf(prevalence: np.ndarray, relative_risk: float) -> np.ndarray:
    return prevalence * (relative_risk - 1) / (1 + prevalence * (relative_risk - 1))


def run_reconstruction(gbd: pd.DataFrame, gco: pd.DataFrame, figure: FigureSurface, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    gbd = gbd.copy()
    gbd["prevalence_age"] = gbd["age_midpoint"] - 10
    gbd_ages = gbd["age_midpoint"].to_numpy()
    gbd_cases = gbd["val"].to_numpy(dtype=float)
    gbd_old = gbd_ages >= 75
    gco_cases = gco["cases"].to_numpy(dtype=float)
    gco_ages = gco["age_midpoint"].to_numpy(dtype=float)
    methods = ["median_patch", "nearest", "bilinear"]
    rr_noncardia = {"lower": 3.25, "point": 5.94, "upper": 10.86}
    rr_cardia = {"lower": 1.54, "point": 3.06, "upper": 6.10}
    summary_rows, age_rows, sensitivity_rows = [], [], []
    for method, multiplier, n_label, c_label, older_method, fraction in itertools.product(
        methods, [1.0, 0.95, 1.05], rr_noncardia, rr_cardia,
        ["gbd_weighted_prevalence", "midpoint", "gbd_weighted_paf"], [0.5, 0.7]
    ):
        base_prevalence = np.array([figure.prevalence(age, method=method) for age in gbd["prevalence_age"]])
        p = np.clip(base_prevalence * multiplier, 0, 1)
        p75 = figure.prevalence(72.5, method=method) * multiplier
        if older_method == "midpoint":
            gco_prevalence = np.r_[p[~gbd_old], np.clip(p75, 0, 1)]
        else:
            weighted = float(np.average(p[gbd_old], weights=gbd_cases[gbd_old]))
            gco_prevalence = np.r_[p[~gbd_old], weighted]
        rn, rc = rr_noncardia[n_label], rr_cardia[c_label]
        gbd_paf = fraction * paf(p, rn) + (1 - fraction) * paf(p, rc)
        gco_paf = fraction * paf(gco_prevalence, rn) + (1 - fraction) * paf(gco_prevalence, rc)
        if older_method == "gbd_weighted_paf":
            gco_paf[-1] = float(np.average(gbd_paf[gbd_old], weights=gbd_cases[gbd_old]))
        gbd_attributable = float(gbd_cases @ gbd_paf)
        gco_attributable = float(gco_cases @ gco_paf)
        result = {
            "method": method, "prevalence_multiplier": multiplier,
            "rr_noncardia": rn, "rr_cardia": rc, "older_method": older_method,
            "noncardia_fraction": fraction, "gbd_attributable": gbd_attributable,
            "globocan_attributable": gco_attributable,
            "reduction_percent": 100 * (1 - gco_attributable / gbd_attributable),
        }
        sensitivity_rows.append(result)
        if method == "median_patch" and multiplier == 1 and n_label == "point" and c_label == "point" and older_method == "gbd_weighted_prevalence":
            summary_rows.append(result)
            for source, groups, prev, inc, pfs in [
                ("GBD", gbd["age_name"].tolist(), p, gbd_cases, gbd_paf),
                ("GLOBOCAN", gco["age_group"].tolist(), gco_prevalence, gco_cases, gco_paf),
            ]:
                for group, prevalence_value, incidence, fraction_value in zip(groups, prev, inc, pfs):
                    age_rows.append({"source": source, "age_group": group, "noncardia_fraction": fraction,
                                     "prevalence": prevalence_value, "paf": fraction_value,
                                     "incidence_cases": incidence, "attributable_cases": incidence * fraction_value})
    summary = pd.DataFrame(summary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    age_details = pd.DataFrame(age_rows)
    summary.to_csv(output / "primary_results.csv", index=False)
    sensitivity.to_csv(output / "deterministic_sensitivity.csv", index=False)
    age_details.to_csv(output / "age_specific_details.csv", index=False)
    return summary, sensitivity


def main() -> None:
    args = parse_args()
    for path in (args.gbd, args.globocan, args.figure_pdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    gbd = load_gbd(args.gbd)
    gco = load_globocan(args.globocan)
    figure = FigureSurface(args.figure_pdf, args.page)
    contour = figure.contour_check()
    contour.to_csv(args.output / "contour_internal_check.csv", index=False)
    summary, sensitivity = run_reconstruction(gbd, gco, figure, args.output)
    audit = figure.audit(contour)
    audit.update({
        "gbd_denominator_age20plus": float(gbd["val"].sum()),
        "globocan_denominator_age20plus": float(gco["cases"].sum()),
        "denominator_ratio": float(gbd["val"].sum() / gco["cases"].sum()),
        "aggregate_reduction_percent": float(100 * (1 - gco["cases"].sum() / gbd["val"].sum())),
        "sensitivity_scenarios": int(len(sensitivity)),
        "deterministic_reduction_percent_min": float(sensitivity["reduction_percent"].min()),
        "deterministic_reduction_percent_max": float(sensitivity["reduction_percent"].max()),
    })
    (args.output / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nDeterministic reduction range: {audit['deterministic_reduction_percent_min']:.4f}%–{audit['deterministic_reduction_percent_max']:.4f}%")


if __name__ == "__main__":
    main()
