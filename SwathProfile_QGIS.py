# -*- coding: utf-8 -*-
"""
SwathProfile_QGIS.py
--------------------
QGIS Processing algorithm for generating swath profiles along a centerline.

Workflow
--------
1. Divide the input centerline into non-overlapping perpendicular cross-section
   rectangles at the requested sample interval.  Each rectangle is
   (sample_interval) long along the profile and (2 × swath_half_width) wide.
2. Run QgsZonalStatistics (qgis.analysis) on those rectangles against the DEM
   to compute Min, Max, Mean, Median, and Std Dev per slice.
3. Write results to:
   - A CSV table (distance along profile + per-slice statistics)
   - A polygon feature layer (the slice rectangles with stats as attributes)

Install
-------
Processing Toolbox → Scripts → Open Existing Script… → browse to this file.
The tool will then appear under Processing Toolbox → Scripts → Swath Profile.

Requirements
------------
QGIS 3.16 or later.  No extra extensions needed — QgsZonalStatistics is part
of the standard qgis.analysis module bundled with every QGIS installation.
The input line must be in a projected CRS so that distance and width values
are meaningful (metres, feet, etc.).
"""

import csv
import math

from qgis.analysis import QgsZonalStatistics
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


class SwathProfileAlgorithm(QgsProcessingAlgorithm):
    """Swath profile along a centerline using zonal statistics on a DEM."""

    INPUT_LINE      = "INPUT_LINE"
    INPUT_DEM       = "INPUT_DEM"
    SWATH_WIDTH     = "SWATH_WIDTH"
    SAMPLE_INTERVAL = "SAMPLE_INTERVAL"
    OUTPUT_CSV      = "OUTPUT_CSV"
    OUTPUT_LAYER    = "OUTPUT_LAYER"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def name(self):
        return "swathprofile"

    def displayName(self):
        return "Swath Profile"

    def group(self):
        return "Terrain Analysis"

    def groupId(self):
        return "terrainanalysis"

    def shortHelpString(self):
        return (
            "Generates a swath profile along a centerline polyline.\n\n"
            "The line is divided into non-overlapping perpendicular cross-section "
            "rectangles at the requested sample interval. Elevation statistics "
            "(Min, Max, Mean, Median, Std Dev) are computed for each slice from "
            "a DEM using QgsZonalStatistics.\n\n"
            "Outputs:\n"
            "  \u2022 CSV table \u2013 distance along profile + per-slice statistics\n"
            "  \u2022 Polygon layer \u2013 slice rectangles with stats as attributes\n\n"
            "Note: use a projected CRS (metres/feet) so that the half-width and "
            "sample interval are in meaningful map units."
        )

    def createInstance(self):
        return SwathProfileAlgorithm()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_LINE,
                "Profile Centerline",
                [QgsProcessing.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                "DEM Raster",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SWATH_WIDTH,
                "Swath Half-Width (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0,
                minValue=0.001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLE_INTERVAL,
                "Sample Interval (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.001,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LAYER,
                "Output Slice Polygons",
                type=QgsProcessing.TypeVectorPolygon,
            )
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        source          = self.parameterAsSource(parameters, self.INPUT_LINE, context)
        dem             = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        swath_width     = self.parameterAsDouble(parameters, self.SWATH_WIDTH, context)
        sample_interval = self.parameterAsDouble(parameters, self.SAMPLE_INTERVAL, context)
        out_csv_path    = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        crs = source.sourceCrs()

        # ---- Read input line ----------------------------------------
        feats = list(source.getFeatures())
        if not feats:
            raise QgsProcessingException("No features found in the input line layer.")

        line_geom    = feats[0].geometry()
        total_length = line_geom.length()
        feedback.pushInfo(f"Line length: {total_length:.2f} map units")

        if sample_interval >= total_length:
            feedback.pushWarning(
                "Sample interval is >= line length. Only one slice will be produced."
            )

        # ---- Generate slice centre distances ------------------------
        # Centers at half, 3/2, 5/2 … × interval → perfect non-overlapping
        # tiling with no gap at the start.  A partial slice at the end is
        # intentionally omitted to avoid overlap.
        half     = sample_interval / 2.0
        n_slices = max(1, int(total_length // sample_interval))
        centers  = [half + i * sample_interval for i in range(n_slices)]
        feedback.pushInfo(f"Generating {n_slices} cross-section slices…")

        # ---- Helper: perpendicular slice rectangle ------------------
        def make_slice(geom, dist, half_width, half_len):
            """
            Return a QgsGeometry polygon perpendicular to *geom* at *dist*.

            The rectangle spans ±half_len along the line direction and
            ±half_width perpendicular to it.
            """
            pt = geom.interpolate(dist).asPoint()
            cx, cy = pt.x(), pt.y()

            # Estimate tangent from two nearby points
            d_step = max(sample_interval / 100.0, 1e-3)
            p0 = geom.interpolate(max(0.0, dist - d_step)).asPoint()
            p1 = geom.interpolate(min(geom.length(), dist + d_step)).asPoint()

            ax = p1.x() - p0.x()
            ay = p1.y() - p0.y()
            mag = math.hypot(ax, ay)
            if mag < 1e-10:
                return None
            ax /= mag
            ay /= mag
            px, py = -ay, ax  # rotate 90° CCW → perpendicular

            corners = [
                QgsPointXY(cx - half_len * ax + half_width * px,
                           cy - half_len * ay + half_width * py),
                QgsPointXY(cx + half_len * ax + half_width * px,
                           cy + half_len * ay + half_width * py),
                QgsPointXY(cx + half_len * ax - half_width * px,
                           cy + half_len * ay - half_width * py),
                QgsPointXY(cx - half_len * ax - half_width * px,
                           cy - half_len * ay - half_width * py),
            ]
            return QgsGeometry.fromPolygonXY([corners])

        # ---- Build in-memory zone layer -----------------------------
        zone_fields = QgsFields()
        zone_fields.append(QgsField("SLICE_ID", QVariant.Int))
        zone_fields.append(QgsField("DIST_M",   QVariant.Double))

        mem_uri   = f"Polygon?crs={crs.authid()}"
        mem_layer = QgsVectorLayer(mem_uri, "swath_zones", "memory")
        mem_layer.dataProvider().addAttributes(zone_fields)
        mem_layer.updateFields()

        zone_feats = []
        for i, dist in enumerate(centers):
            if feedback.isCanceled():
                return {}
            g = make_slice(line_geom, dist, swath_width, half)
            if g is None:
                feedback.pushWarning(f"Slice {i} at dist {dist:.2f} skipped (degenerate geometry).")
                continue
            f = QgsFeature(mem_layer.fields())
            f.setGeometry(g)
            f["SLICE_ID"] = i
            f["DIST_M"]   = dist
            zone_feats.append(f)

        mem_layer.dataProvider().addFeatures(zone_feats)
        mem_layer.updateExtents()

        # ---- Zonal statistics ---------------------------------------
        feedback.pushInfo("Computing zonal statistics…")

        stat_flags = (
            QgsZonalStatistics.Min    |
            QgsZonalStatistics.Max    |
            QgsZonalStatistics.Mean   |
            QgsZonalStatistics.Median |
            QgsZonalStatistics.StDev
        )
        zs = QgsZonalStatistics(
            mem_layer,          # polygon layer (modified in-place)
            dem,                # raster layer
            "_",                # field prefix
            1,                  # raster band
            stat_flags,
        )
        zs.calculateStatistics(feedback)

        # Map the generated field names (prefix + stat name) to our output names.
        # QgsZonalStatistics uses lowercase: _min, _max, _mean, _median, _stdev
        zonal_to_out = {
            "_min":    "MIN",
            "_max":    "MAX",
            "_mean":   "MEAN",
            "_median": "MEDIAN",
            "_stdev":  "STD",
        }
        avail_fields = {f.name() for f in mem_layer.fields()}
        active_map   = {k: v for k, v in zonal_to_out.items() if k in avail_fields}

        if not active_map:
            feedback.reportError(
                "QgsZonalStatistics did not add any statistic fields.  "
                "Check that the DEM overlaps the profile line and that "
                "the raster and line are in the same CRS.",
                fatalError=False,
            )

        # ---- Define output feature fields ---------------------------
        stat_out_names = ["MIN", "MAX", "MEAN", "MEDIAN", "STD"]
        out_fields = QgsFields()
        out_fields.append(QgsField("SLICE_ID", QVariant.Int))
        out_fields.append(QgsField("DIST_M",   QVariant.Double))
        for name in stat_out_names:
            out_fields.append(QgsField(name, QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT_LAYER, context,
            out_fields, QgsWkbTypes.Polygon, crs,
        )

        # ---- Write output features + collect CSV rows ---------------
        results = []
        for feat in mem_layer.getFeatures():
            if feedback.isCanceled():
                return {}
            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(feat.geometry())
            out_feat["SLICE_ID"] = feat["SLICE_ID"]
            out_feat["DIST_M"]   = feat["DIST_M"]

            rec = {
                "SLICE_ID": feat["SLICE_ID"],
                "DIST_M":   feat["DIST_M"],
            }
            for zfield, out_name in active_map.items():
                val = feat[zfield]
                out_feat[out_name] = val
                rec[out_name] = val

            sink.addFeature(out_feat)
            results.append(rec)

        results.sort(key=lambda r: (r.get("DIST_M") or 0.0))

        # ---- Write CSV ----------------------------------------------
        csv_cols = ["SLICE_ID", "DIST_M"] + stat_out_names
        with open(out_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=csv_cols, extrasaction="ignore", restval=""
            )
            writer.writeheader()
            writer.writerows(results)

        feedback.pushInfo(f"CSV written: {out_csv_path}")
        feedback.pushInfo(
            f"Done \u2014 {len(results)} slices along a {total_length:.1f}-unit line."
        )

        return {
            self.OUTPUT_CSV:   out_csv_path,
            self.OUTPUT_LAYER: dest_id,
        }
