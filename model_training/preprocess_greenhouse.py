#!/usr/bin/env python3
"""Validate timestamped greenhouse files and derive forecast-ready features."""

import argparse
import csv
import glob
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MINUTE = timedelta(minutes=1)
WINDOW_MINUTES = 15
FORECAST_HORIZON = timedelta(minutes=30)
MAGNUS_A = 17.27
MAGNUS_B = 237.7


class PreprocessingError(Exception):
    """Raised when an input cannot be safely used for temporal training."""


@dataclass(frozen=True)
class SensorRecord:
    timestamp: datetime
    temperature_c: float
    relative_humidity_pct: float


@dataclass
class ReadCounts:
    rows: int = 0
    missing_measurements: int = 0
    invalid_measurements: int = 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine compatible timestamped CSV/XLSX files, calculate "
            "environmental features, and create exact +30-minute targets."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV/XLSX file paths or glob patterns. Files are read locally only.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--temperature-column", default="temperature")
    parser.add_argument("--humidity-column", default="humidity")
    parser.add_argument("--temperature-unit", choices=("C", "F"), required=True)
    parser.add_argument(
        "--humidity-unit", choices=("percent", "fraction"), required=True
    )
    parser.add_argument(
        "--timestamp-timezone",
        choices=("utc", "unknown"),
        default="utc",
        help=(
            "Use utc only for timestamps with documented offsets; use unknown "
            "to preserve naive timestamps without conversion."
        ),
    )
    parser.add_argument(
        "--sheet-name",
        default="Data",
        help="Worksheet to read from XLSX inputs.",
    )
    parser.add_argument(
        "--sample-interval-minutes",
        type=int,
        default=1,
        help="Expected interval for complete 15-minute input windows.",
    )
    parser.add_argument(
        "--resample-one-minute",
        action="store_true",
        help=(
            "Explicitly aggregate higher-frequency observations into occupied "
            "one-minute bins. No empty bins are filled."
        ),
    )
    return parser.parse_args(argv)


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        if not matches:
            raise PreprocessingError("No input file matched: {}".format(pattern))
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise PreprocessingError("Input is not a file: {}".format(path))
            if path.suffix.lower() not in (".csv", ".xlsx"):
                raise PreprocessingError(
                    "Input is not a CSV or XLSX file: {}".format(path)
                )
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


def parse_timestamp(
    value: object,
    path: Path,
    line_number: int,
    timestamp_timezone: str,
    excel_epoch: Optional[datetime] = None,
) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        if excel_epoch is None:
            raise PreprocessingError(
                "{}:{} has a numeric timestamp without an Excel date system.".format(
                    path, line_number
                )
            )
        try:
            from openpyxl.utils.datetime import from_excel

            parsed = from_excel(value, epoch=excel_epoch)
        except (OverflowError, TypeError, ValueError) as error:
            raise PreprocessingError(
                "{}:{} has an invalid Excel timestamp.".format(path, line_number)
            ) from error
        if not isinstance(parsed, datetime):
            raise PreprocessingError(
                "{}:{} Excel value is not a full date and time.".format(
                    path, line_number
                )
            )
    else:
        text = str(value or "").strip()
        if not text:
            raise PreprocessingError(
                "{}:{} has an empty timestamp.".format(path, line_number)
            )
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as iso_error:
            parsed = None
            for timestamp_format in (
                "%d-%m-%Y %H:%M:%S.%f",
                "%d-%m-%Y %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(text, timestamp_format)
                    break
                except ValueError:
                    continue
            if parsed is None:
                raise PreprocessingError(
                    "{}:{} has an invalid timestamp {!r}.".format(
                        path, line_number, value
                    )
                ) from iso_error

    has_offset = parsed.tzinfo is not None and parsed.utcoffset() is not None
    if timestamp_timezone == "unknown":
        if has_offset:
            raise PreprocessingError(
                "{}:{} has an explicit timezone offset, but timezone mode is "
                "'unknown'. Choose UTC mode only if the source timezone is "
                "documented.".format(path, line_number)
            )
        return parsed.replace(tzinfo=None)
    if not has_offset:
        raise PreprocessingError(
            "{}:{} timestamp has no documented UTC offset; use "
            "--timestamp-timezone unknown to preserve it without conversion."
            .format(path, line_number)
        )
    return parsed.astimezone(timezone.utc)


def convert_temperature(value: float, unit: str) -> float:
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    return value


def convert_humidity(value: float, unit: str) -> float:
    if unit == "fraction":
        return value * 100.0
    return value


def resolve_columns(
    fieldnames: Sequence[object],
    requested: Dict[str, str],
    path: Path,
) -> Dict[str, int]:
    names = [str(name).strip() if name is not None else "" for name in fieldnames]
    normalized = [name.casefold() for name in names if name]
    if len(normalized) != len(set(normalized)):
        raise PreprocessingError("{} has duplicate column names.".format(path))
    lookup = {
        name.casefold(): index for index, name in enumerate(names) if name
    }
    missing_columns = [
        name for name in requested.values() if name.casefold() not in lookup
    ]
    if missing_columns:
        raise PreprocessingError(
            "{} is missing required column(s): {}. Found: {}.".format(
                path, ", ".join(missing_columns), ", ".join(names)
            )
        )
    return {
        key: lookup[name.casefold()] for key, name in requested.items()
    }


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def parse_measurement(
    timestamp: datetime,
    temperature_value: object,
    humidity_value: object,
    path: Path,
    line_number: int,
    temperature_unit: str,
    humidity_unit: str,
    counts: ReadCounts,
) -> Optional[SensorRecord]:
    counts.rows += 1
    if is_missing(temperature_value) or is_missing(humidity_value):
        counts.missing_measurements += 1
        return None
    try:
        temperature = convert_temperature(
            float(temperature_value), temperature_unit
        )
        humidity = convert_humidity(float(humidity_value), humidity_unit)
    except (TypeError, ValueError) as error:
        raise PreprocessingError(
            "{}:{} has a non-numeric temperature or humidity.".format(
                path, line_number
            )
        ) from error
    if (
        not math.isfinite(temperature)
        or not math.isfinite(humidity)
        or temperature < -40.0
        or temperature > 80.0
        or humidity <= 0.0
        or humidity > 100.0
    ):
        counts.invalid_measurements += 1
        return None
    return SensorRecord(timestamp, temperature, humidity)


def read_csv_file(
    path: Path,
    timestamp_column: str,
    temperature_column: str,
    humidity_column: str,
    temperature_unit: str,
    humidity_unit: str,
    timestamp_timezone: str,
) -> Tuple[List[SensorRecord], ReadCounts]:
    records: List[SensorRecord] = []
    counts = ReadCounts()
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise PreprocessingError("{} has no CSV header.".format(path))
        fieldnames = [name.strip() for name in reader.fieldnames]
        column_indices = resolve_columns(
            fieldnames,
            {
                "timestamp": timestamp_column,
                "temperature": temperature_column,
                "humidity": humidity_column,
            },
            path,
        )
        reader.fieldnames = fieldnames
        timestamp_key = fieldnames[column_indices["timestamp"]]
        temperature_key = fieldnames[column_indices["temperature"]]
        humidity_key = fieldnames[column_indices["humidity"]]

        for line_number, row in enumerate(reader, start=2):
            if not row or all(is_missing(value) for value in row.values()):
                continue
            if None in row and row[None]:
                raise PreprocessingError(
                    "{}:{} has more values than the CSV header.".format(
                        path, line_number
                    )
                )
            timestamp = parse_timestamp(
                row.get(timestamp_key), path, line_number, timestamp_timezone
            )
            record = parse_measurement(
                timestamp,
                row.get(temperature_key),
                row.get(humidity_key),
                path,
                line_number,
                temperature_unit,
                humidity_unit,
                counts,
            )
            if record is not None:
                records.append(record)
    return records, counts


def read_xlsx_file(
    path: Path,
    timestamp_column: str,
    temperature_column: str,
    humidity_column: str,
    temperature_unit: str,
    humidity_unit: str,
    timestamp_timezone: str,
    sheet_name: str,
) -> Tuple[List[SensorRecord], ReadCounts]:
    try:
        from openpyxl import load_workbook
        from openpyxl.utils.exceptions import InvalidFileException
        from zipfile import BadZipFile
    except ImportError as error:
        raise PreprocessingError(
            "XLSX input requires openpyxl; install model_training/requirements.txt."
        ) from error

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException) as error:
        raise PreprocessingError(
            "{} is not a valid XLSX workbook.".format(path)
        ) from error

    records: List[SensorRecord] = []
    counts = ReadCounts()
    try:
        if sheet_name not in workbook.sheetnames:
            raise PreprocessingError(
                "{} has no worksheet {!r}. Found: {}.".format(
                    path, sheet_name, ", ".join(workbook.sheetnames)
                )
            )
        worksheet = workbook[sheet_name]
        header = next(
            worksheet.iter_rows(min_row=1, max_row=1, values_only=True), None
        )
        if not header:
            raise PreprocessingError(
                "{} worksheet {!r} has no header.".format(path, sheet_name)
            )
        column_indices = resolve_columns(
            header,
            {
                "timestamp": timestamp_column,
                "temperature": temperature_column,
                "humidity": humidity_column,
            },
            path,
        )

        for line_number, row in enumerate(
            worksheet.iter_rows(min_row=2, values_only=True), start=2
        ):
            if all(is_missing(value) for value in row):
                continue
            timestamp = parse_timestamp(
                row[column_indices["timestamp"]],
                path,
                line_number,
                timestamp_timezone,
                workbook.epoch,
            )
            record = parse_measurement(
                timestamp,
                row[column_indices["temperature"]],
                row[column_indices["humidity"]],
                path,
                line_number,
                temperature_unit,
                humidity_unit,
                counts,
            )
            if record is not None:
                records.append(record)
    finally:
        workbook.close()
    return records, counts


def read_sensor_file(
    path: Path,
    timestamp_column: str,
    temperature_column: str,
    humidity_column: str,
    temperature_unit: str,
    humidity_unit: str,
    timestamp_timezone: str,
    sheet_name: str,
) -> Tuple[List[SensorRecord], ReadCounts]:
    if path.suffix.lower() == ".xlsx":
        return read_xlsx_file(
            path,
            timestamp_column,
            temperature_column,
            humidity_column,
            temperature_unit,
            humidity_unit,
            timestamp_timezone,
            sheet_name,
        )
    return read_csv_file(
        path,
        timestamp_column,
        temperature_column,
        humidity_column,
        temperature_unit,
        humidity_unit,
        timestamp_timezone,
    )


def combine_records(
    files: Sequence[Path],
    timestamp_column: str,
    temperature_column: str,
    humidity_column: str,
    temperature_unit: str,
    humidity_unit: str,
    timestamp_timezone: str,
    sheet_name: str,
) -> Tuple[List[SensorRecord], ReadCounts, int]:
    by_timestamp: Dict[datetime, SensorRecord] = {}
    totals = ReadCounts()
    duplicate_rows = 0
    for path in files:
        records, counts = read_sensor_file(
            path,
            timestamp_column,
            temperature_column,
            humidity_column,
            temperature_unit,
            humidity_unit,
            timestamp_timezone,
            sheet_name,
        )
        totals.rows += counts.rows
        totals.missing_measurements += counts.missing_measurements
        totals.invalid_measurements += counts.invalid_measurements
        for record in records:
            previous = by_timestamp.get(record.timestamp)
            if previous is None:
                by_timestamp[record.timestamp] = record
            elif (
                previous.temperature_c == record.temperature_c
                and previous.relative_humidity_pct == record.relative_humidity_pct
            ):
                duplicate_rows += 1
            else:
                raise PreprocessingError(
                    "Conflicting readings share timestamp {}.".format(
                        record.timestamp.isoformat()
                    )
                )
    combined = [
        by_timestamp[timestamp] for timestamp in sorted(by_timestamp.keys())
    ]
    if not combined:
        raise PreprocessingError("No valid timestamped measurements remain.")
    return combined, totals, duplicate_rows


def resample_one_minute(records: Sequence[SensorRecord]) -> List[SensorRecord]:
    if len(records) < 2:
        raise PreprocessingError(
            "Cannot justify one-minute aggregation with fewer than two readings."
        )
    intervals = [
        (records[index].timestamp - records[index - 1].timestamp).total_seconds()
        for index in range(1, len(records))
    ]
    if statistics.median(intervals) > MINUTE.total_seconds():
        raise PreprocessingError(
            "One-minute aggregation refused: median source interval exceeds one "
            "minute. Upsampling slower data would invent measurements."
        )

    bins: Dict[datetime, List[SensorRecord]] = {}
    for record in records:
        minute = record.timestamp.replace(second=0, microsecond=0)
        bins.setdefault(minute, []).append(record)
    result = []
    for timestamp in sorted(bins):
        readings = bins[timestamp]
        result.append(
            SensorRecord(
                timestamp,
                sum(item.temperature_c for item in readings) / len(readings),
                sum(item.relative_humidity_pct for item in readings) / len(readings),
            )
        )
    return result


def calculate_features(record: SensorRecord) -> Tuple[float, float, float]:
    alpha = (
        math.log(record.relative_humidity_pct / 100.0)
        + MAGNUS_A * record.temperature_c / (MAGNUS_B + record.temperature_c)
    )
    dew_point_c = MAGNUS_B * alpha / (MAGNUS_A - alpha)
    saturation_vapor_pressure_kpa = 0.6108 * math.exp(
        MAGNUS_A * record.temperature_c / (MAGNUS_B + record.temperature_c)
    )
    vpd_kpa = saturation_vapor_pressure_kpa * (
        1.0 - record.relative_humidity_pct / 100.0
    )
    if not all(
        math.isfinite(value)
        for value in (dew_point_c, vpd_kpa, record.temperature_c - dew_point_c)
    ):
        raise PreprocessingError(
            "Feature calculation produced a non-finite value at {}.".format(
                record.timestamp.isoformat()
            )
        )
    return dew_point_c, vpd_kpa, record.temperature_c - dew_point_c


def write_processed_csv(
    path: Path,
    records: Sequence[SensorRecord],
    timestamp_timezone: str,
    sample_interval_minutes: int,
) -> int:
    if path.exists():
        raise PreprocessingError(
            "Output already exists; choose a new path to avoid overwriting data."
        )
    if sample_interval_minutes <= 0 or WINDOW_MINUTES % sample_interval_minutes:
        raise PreprocessingError(
            "The sample interval must be a positive divisor of the 15-minute window."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    by_timestamp = {record.timestamp: record for record in records}
    timestamp_column = (
        "timestamp_utc" if timestamp_timezone == "utc" else "timestamp"
    )
    target_timestamp_column = (
        "forecast_target_timestamp_utc"
        if timestamp_timezone == "utc"
        else "forecast_target_timestamp"
    )
    fieldnames = (
        timestamp_column,
        "timestamp_timezone",
        "sample_interval_minutes",
        "temperature_c",
        "relative_humidity_pct",
        "dew_point_c",
        "vpd_kpa",
        "dew_point_margin_c",
        "forecast_window_15min_complete",
        target_timestamp_column,
        "target_temperature_30m_c",
        "target_relative_humidity_30m_pct",
    )
    forecast_examples = 0
    consecutive_samples = 0
    previous_timestamp = None
    sample_interval = timedelta(minutes=sample_interval_minutes)
    window_samples = WINDOW_MINUTES // sample_interval_minutes + 1
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if previous_timestamp is not None and (
                record.timestamp - previous_timestamp == sample_interval
            ):
                consecutive_samples += 1
            else:
                consecutive_samples = 1
            previous_timestamp = record.timestamp

            window_complete = consecutive_samples >= window_samples
            target_timestamp = record.timestamp + FORECAST_HORIZON
            target = (
                by_timestamp.get(target_timestamp) if window_complete else None
            )
            dew_point_c, vpd_kpa, dew_point_margin_c = calculate_features(record)
            if target is not None:
                forecast_examples += 1
            writer.writerow(
                {
                    timestamp_column: record.timestamp.isoformat(),
                    "timestamp_timezone": (
                        "UTC" if timestamp_timezone == "utc" else "unknown"
                    ),
                    "sample_interval_minutes": sample_interval_minutes,
                    "temperature_c": record.temperature_c,
                    "relative_humidity_pct": record.relative_humidity_pct,
                    "dew_point_c": dew_point_c,
                    "vpd_kpa": vpd_kpa,
                    "dew_point_margin_c": dew_point_margin_c,
                    "forecast_window_15min_complete": str(window_complete).lower(),
                    target_timestamp_column: target_timestamp.isoformat(),
                    "target_temperature_30m_c": (
                        target.temperature_c if target is not None else ""
                    ),
                    "target_relative_humidity_30m_pct": (
                        target.relative_humidity_pct if target is not None else ""
                    ),
                }
            )
    return forecast_examples


def run(args: argparse.Namespace) -> None:
    if args.sample_interval_minutes <= 0:
        raise PreprocessingError(
            "--sample-interval-minutes must be a positive integer."
        )
    if WINDOW_MINUTES % args.sample_interval_minutes:
        raise PreprocessingError(
            "--sample-interval-minutes must divide the 15-minute input window."
        )
    if args.resample_one_minute and args.sample_interval_minutes != 1:
        raise PreprocessingError(
            "--resample-one-minute requires --sample-interval-minutes 1."
        )
    files = expand_inputs(args.inputs)
    output_path = args.output.resolve()
    if output_path in files:
        raise PreprocessingError("Output path must not overwrite an input file.")

    records, counts, duplicate_rows = combine_records(
        files,
        args.timestamp_column,
        args.temperature_column,
        args.humidity_column,
        args.temperature_unit,
        args.humidity_unit,
        args.timestamp_timezone,
        args.sheet_name,
    )
    if args.resample_one_minute:
        records = resample_one_minute(records)
    forecast_examples = write_processed_csv(
        output_path,
        records,
        args.timestamp_timezone,
        args.sample_interval_minutes,
    )

    print("Input files: {}".format(len(files)))
    print("Source rows with timestamps: {}".format(counts.rows))
    print("Valid processed readings: {}".format(len(records)))
    print("Rows with missing measurements dropped: {}".format(counts.missing_measurements))
    print("Rows with invalid measurements dropped: {}".format(counts.invalid_measurements))
    print("Exact duplicate timestamps dropped: {}".format(duplicate_rows))
    print("Sample interval: {} minutes".format(args.sample_interval_minutes))
    print("Timestamp timezone: {}".format(args.timestamp_timezone))
    print("No timezone conversion was applied to unknown timestamps.")
    print(
        "Rows with a complete 15-minute input window and exact +30-minute targets: "
        "{}".format(forecast_examples)
    )
    print("No missing timestamps or measurements were filled.")
    print("Processed CSV: {}".format(output_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        run(args)
    except PreprocessingError as error:
        print("preprocessing error: {}".format(error), file=sys.stderr)
        return 2
    except OSError as error:
        print("file error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
