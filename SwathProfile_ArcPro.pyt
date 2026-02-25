"""
SwathProfile_ArcPro.pyt
-----------------------
ArcGIS Pro Python Toolbox for generating swath profiles along a centerline.

Workflow
--------
1. Divide the input centerline into perpendicular cross-section rectangles at
   the requested sample interval.  Each rectangle is (sample_interval) long
   along the profile and (2 × swath_half_width) wide across it.
2. Run Zonal Statistics as Table (Spatial Analyst) using those rectangles as
   zones against the input DEM.
3. Write results to:
   - A CSV table (distance along profile + Min / Max / Mean / Median / Std Dev)
   - A polygon feature class (the slice rectangles with stats as attributes)

Requirements
------------
- ArcGIS Pro with the Spatial Analyst extension.
- Input line must be in a projected coordinate system (map units = metres or
  feet) so that distance and width values are meaningful.
"""

import arcpy
import csv
import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# Toolbox definition
# ---------------------------------------------------------------------------

class Toolbox:
    def __init__(self):
        self.label = "Swath Profile Tools"
        self.alias = "swathprofile"
        self.tools = [SwathProfile]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class SwathProfile:
    def __init__(self):
        self.label = "Swath Profile"
        self.description = (
            "Generate a swath profile along a centerline.  The line is divided "
            "into perpendicular cross-section slices at a specified interval.  "
            "Elevation statistics (Min, Max, Mean, Median, Std Dev) are computed "
            "for each slice from a DEM using the Spatial Analyst extension."
        )
        self.canRunInBackground = False

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        p_line = arcpy.Parameter(
            displayName="Profile Centerline",
            name="in_line",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        p_line.filter.list = ["Polyline"]

        p_dem = arcpy.Parameter(
            displayName="DEM Raster",
            name="in_dem",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        p_width = arcpy.Parameter(
            displayName="Swath Half-Width (map units)",
            name="swath_width",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        p_width.value = 500.0

        p_interval = arcpy.Parameter(
            displayName="Sample Interval (map units)",
            name="sample_interval",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        p_interval.value = 100.0

        p_csv = arcpy.Parameter(
            displayName="Output CSV",
            name="out_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Output",
        )
        p_csv.filter.list = ["csv"]

        p_fc = arcpy.Parameter(
            displayName="Output Slice Polygons (Feature Class)",
            name="out_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        return [p_line, p_dem, p_width, p_interval, p_csv, p_fc]

    def isLicensed(self):
        return arcpy.CheckExtension("Spatial") == "Available"

    def updateMessages(self, parameters):
        return

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        in_line       = parameters[0].valueAsText
        in_dem        = parameters[1].valueAsText
        swath_width   = float(parameters[2].value)
        sample_interval = float(parameters[3].value)
        out_csv_path  = parameters[4].valueAsText
        out_fc        = parameters[5].valueAsText

        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True

        sr = arcpy.Describe(in_line).spatialReference

        # ---- Read input line ----------------------------------------
        line_geom = None
        with arcpy.da.SearchCursor(in_line, ["SHAPE@"]) as cur:
            for row in cur:
                line_geom = row[0]
                break

        if line_geom is None:
            messages.addErrorMessage("No features found in the input line layer.")
            raise arcpy.ExecuteError

        total_length = line_geom.length
        messages.addMessage(f"Line length: {total_length:.2f} map units")

        if sample_interval >= total_length:
            messages.addWarningMessage(
                "Sample interval is >= line length. Only one slice will be produced."
            )

        # ---- Build slice centre distances ----------------------------
        # Centers are placed at half, 3/2, 5/2, ... of the interval so
        # that each rectangle [center - half, center + half] tiles the
        # line perfectly without overlap.
        half = sample_interval / 2.0
        centers = list(np.arange(half, total_length, sample_interval))

        if not centers:
            centers = [total_length / 2.0]

        messages.addMessage(f"Generating {len(centers)} cross-section slices...")

        # ---- Helper: build perpendicular rectangle at distance -------
        def make_slice_polygon(geom, dist, half_width, half_len, sr):
            """
            Return a Polygon perpendicular to *geom* at *dist* along the line.

            The rectangle spans ±half_len along the line direction and
            ±half_width perpendicular to it.
            """
            cx = geom.positionAlongLine(dist).firstPoint.X
            cy = geom.positionAlongLine(dist).firstPoint.Y

            # Estimate tangent direction from two nearby points
            d_step = max(sample_interval / 100.0, 1e-3)
            d0 = max(0.0, dist - d_step)
            d1 = min(geom.length, dist + d_step)
            p0 = geom.positionAlongLine(d0).firstPoint
            p1 = geom.positionAlongLine(d1).firstPoint

            along_x = p1.X - p0.X
            along_y = p1.Y - p0.Y
            mag = math.hypot(along_x, along_y)
            if mag < 1e-10:
                return None
            along_x /= mag
            along_y /= mag

            # Perpendicular (rotate 90° CCW)
            perp_x, perp_y = -along_y, along_x

            # Four corners of the rectangle
            pts = [
                arcpy.Point(
                    cx - half_len * along_x + half_width * perp_x,
                    cy - half_len * along_y + half_width * perp_y,
                ),
                arcpy.Point(
                    cx + half_len * along_x + half_width * perp_x,
                    cy + half_len * along_y + half_width * perp_y,
                ),
                arcpy.Point(
                    cx + half_len * along_x - half_width * perp_x,
                    cy + half_len * along_y - half_width * perp_y,
                ),
                arcpy.Point(
                    cx - half_len * along_x - half_width * perp_x,
                    cy - half_len * along_y - half_width * perp_y,
                ),
            ]
            arr = arcpy.Array(pts + [pts[0]])  # close ring
            return arcpy.Polygon(arr, sr)

        # ---- Create temporary zone feature class ---------------------
        scratch_gdb = arcpy.env.scratchGDB
        temp_zones = os.path.join(scratch_gdb, "swath_zones_tmp")
        if arcpy.Exists(temp_zones):
            arcpy.management.Delete(temp_zones)

        arcpy.management.CreateFeatureclass(
            os.path.dirname(temp_zones),
            os.path.basename(temp_zones),
            "POLYGON",
            spatial_reference=sr,
        )
        arcpy.management.AddField(temp_zones, "SLICE_ID", "LONG")
        arcpy.management.AddField(temp_zones, "DIST_M",   "DOUBLE")

        with arcpy.da.InsertCursor(temp_zones, ["SHAPE@", "SLICE_ID", "DIST_M"]) as cur:
            for i, dist in enumerate(centers):
                poly = make_slice_polygon(line_geom, dist, swath_width, half, sr)
                if poly is not None:
                    cur.insertRow([poly, i, dist])

        # ---- Zonal Statistics as Table -------------------------------
        zonal_tbl = os.path.join(scratch_gdb, "swath_zonal_tmp")
        if arcpy.Exists(zonal_tbl):
            arcpy.management.Delete(zonal_tbl)

        arcpy.sa.ZonalStatisticsAsTable(
            in_zone_data=temp_zones,
            zone_field="SLICE_ID",
            in_value_raster=in_dem,
            out_table=zonal_tbl,
            statistics_type="ALL",
        )

        # ---- Collect results -----------------------------------------
        dist_map = {}
        with arcpy.da.SearchCursor(temp_zones, ["SLICE_ID", "DIST_M"]) as cur:
            for row in cur:
                dist_map[row[0]] = row[1]

        # Detect which stat fields are actually present in the output table
        # (field names vary slightly between ArcGIS Pro versions)
        avail_fields = {f.name.upper() for f in arcpy.ListFields(zonal_tbl)}
        wanted_stats = ["MIN", "MAX", "MEAN", "STD", "MEDIAN"]
        stat_cols = [f for f in wanted_stats if f in avail_fields]

        if not stat_cols:
            messages.addWarningMessage(
                "No expected statistic fields (MIN/MAX/MEAN/STD/MEDIAN) found in "
                "the zonal statistics table.  Check that the DEM overlaps the "
                "profile line and that Spatial Analyst is licensed."
            )

        read_fields = ["SLICE_ID"] + stat_cols
        results = []
        with arcpy.da.SearchCursor(zonal_tbl, read_fields) as cur:
            for row in cur:
                rec = {
                    "SLICE_ID": row[0],
                    "DIST_M":   dist_map.get(row[0], None),
                }
                for j, col in enumerate(stat_cols, 1):
                    rec[col] = row[j]
                results.append(rec)

        results.sort(key=lambda r: (r["DIST_M"] or 0.0))

        # ---- Write CSV -----------------------------------------------
        csv_cols = ["SLICE_ID", "DIST_M"] + stat_cols
        with open(out_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        messages.addMessage(f"CSV written: {out_csv_path}")

        # ---- Write output feature class ------------------------------
        arcpy.management.CopyFeatures(temp_zones, out_fc)
        for col in stat_cols:
            arcpy.management.AddField(out_fc, col, "DOUBLE")

        stats_lut = {r["SLICE_ID"]: r for r in results}
        update_cols = ["SLICE_ID"] + stat_cols
        with arcpy.da.UpdateCursor(out_fc, update_cols) as cur:
            for row in cur:
                sid = row[0]
                if sid in stats_lut:
                    s = stats_lut[sid]
                    for j, col in enumerate(stat_cols, 1):
                        row[j] = s.get(col)
                    cur.updateRow(row)

        messages.addMessage(f"Slice polygons written: {out_fc}")
        messages.addMessage(
            f"Done — {len(results)} slices along a "
            f"{total_length:.1f}-unit line."
        )

        arcpy.CheckInExtension("Spatial")

    def postExecute(self, parameters):
        return
