#!/usr/bin/env python3
"""
optmz.py - metadata-safe, parallel media optimizer with smart renaming and review.

Usage:
  python optmz.py            # optimize current directory, rename files (default)
  python optmz.py --no-rename
  python optmz.py -r         # review last run (preview + delete prompts)
  python optmz.py -r 2       # review 2 runs ago
  python optmz.py --clean       # delete all original files
  python optmz.py --audit       # check for missing files, auto-fix moved folders
  python optmz.py --fix-paths OLD NEW  # manually fix paths in database
"""
from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party UI: tqdm for progress bar (optional but recommended)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# pillow for EXIF reading (optional but recommended)
try:
    from PIL import ExifTags, Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Constants
OPT_DB_FILE = ".optmzd.db"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ALL_EXTS = IMAGE_EXTS.union(VIDEO_EXTS)
XATTR_PREFIX = "user.optmz."
# MAX_WORKERS = max(1, int((os.cpu_count() or 1) * 0.75))
MAX_WORKERS = 4

# Locate binaries
MAGICK_BIN = shutil.which("magick") or shutil.which("convert")
FFMPEG_BIN = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")


def human_readable(size: int) -> str:
    if size == 0:
        return "0.00 B"
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size / p, 2)
    return f"{s} {'BKBMBGBTB'.replace('B','').split()[i] if False else ['B','KB','MB','GB','TB'][i]}".replace(
        "  ", " "
    )


# --- Extended attributes (xattr) helpers ---
def safe_set_xattr(p: Path, key: str, value: str):
    try:
        os.setxattr(str(p), XATTR_PREFIX + key, value.encode())
    except OSError:
        # ignore on unsupported FS (or fallback), but log
        pass


def safe_get_xattr(p: Path, key: str) -> Optional[str]:
    try:
        v = os.getxattr(str(p), XATTR_PREFIX + key).decode()
        return v
    except OSError:
        return None


# --- SQLite Database Functions ---
def init_db() -> sqlite3.Connection:
    """Initialize SQLite database with optimizations table."""
    conn = sqlite3.connect(OPT_DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            original_path TEXT UNIQUE NOT NULL,
            optimized_path TEXT NOT NULL,
            original_size INTEGER NOT NULL,
            optimized_size INTEGER NOT NULL,
            space_saved INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """
    )
    conn.commit()
    return conn


def is_already_processed(conn: sqlite3.Connection, path: Path) -> bool:
    """Check if a file has already been optimized."""
    # Check xattrs first (most reliable if present)
    xattr_status = safe_get_xattr(path, "status")
    if xattr_status in ("optimized", "no_improvement"):
        return True

    # Check by path or by checking if backup exists with -original-optmz suffix
    resolved = str(path.resolve())
    cursor = conn.execute(
        "SELECT 1 FROM optimizations WHERE original_path = ? OR optimized_path = ?",
        (resolved, resolved),
    )
    if cursor.fetchone():
        return True

    # Also check if this is an optimized file (backup exists)
    backup_name = path.with_name(path.stem + "-original-optmz" + path.suffix)
    if backup_name.exists():
        return True

    # Check if filename matches optimized pattern (DD.MM.YYYY ...)
    if re.match(r"^\d{2}\.\d{2}\.\d{4}", path.stem):
        return True
    return False


def save_optimization(conn: sqlite3.Connection, entry: Dict):
    """Save optimization result immediately to database."""
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO optimizations 
            (timestamp, original_path, optimized_path, original_size, optimized_size, space_saved, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                entry["original"],
                entry["optimized"],
                entry["original_size"],
                entry["optimized_size"],
                entry["space_saved"],
                entry["status"],
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"Database error saving {entry['original']}: {e}")


def get_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> List[Dict]:
    """Get recent optimization runs grouped by timestamp."""
    cursor = conn.execute(
        """
        SELECT timestamp, original_path, optimized_path, original_size, optimized_size, space_saved, status
        FROM optimizations
        ORDER BY timestamp DESC
        """
    )

    runs = defaultdict(list)
    for row in cursor.fetchall():
        ts = row[0].split("T")[0]  # Group by date
        runs[ts].append(
            {
                "timestamp": row[0],
                "original": row[1],
                "optimized": row[2],
                "original_size": row[3],
                "optimized_size": row[4],
                "space_saved": row[5],
                "status": row[6],
            }
        )

    # Convert to list format with correct totals
    result = []
    for ts, files in sorted(runs.items(), reverse=True)[:limit]:
        result.append(
            {
                "timestamp": ts,
                "files": files,
                "total_space_saved": sum(f["space_saved"] for f in files),
            }
        )
    return result


# --- Metadata extraction helpers ---
def _get_exif_from_pillow(p: Path) -> Tuple[Optional[str], Optional[str]]:
    # returns (date_str_iso, device_name)
    if not PIL_AVAILABLE:
        return None, None
    try:
        img = Image.open(str(p))
        exif = img._getexif() or {}
        # map exif tags to names
        exif_named = {}
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            exif_named[tag] = v
        date = exif_named.get("DateTimeOriginal") or exif_named.get("DateTime") or None
        device = exif_named.get("Model") or exif_named.get("Make") or None
        # normalize date to ISO
        if date:
            # common format "YYYY:MM:DD HH:MM:SS"
            try:
                d = datetime.strptime(date, "%Y:%m:%d %H:%M:%S")
                date_iso = d.isoformat()
            except Exception:
                date_iso = None
        else:
            date_iso = None
        if device and isinstance(device, bytes):
            try:
                device = device.decode(errors="ignore")
            except Exception:
                device = str(device)
        return date_iso, device
    except Exception:
        return None, None


def _get_video_metadata_with_ffprobe(p: Path) -> Tuple[Optional[str], Optional[str]]:
    # returns (date_iso, device_name)
    if not FFPROBE_BIN:
        return None, None
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p.resolve()),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out.returncode != 0 or not out.stdout:
            return None, None
        data = json.loads(out.stdout)
        # check format tags
        fmt_tags = (data.get("format") or {}).get("tags") or {}
        creation = fmt_tags.get("creation_time") or fmt_tags.get("encoded_date") or None
        device = (
            fmt_tags.get("encoder")
            or fmt_tags.get("com.apple.quicktime.make")
            or fmt_tags.get("com.apple.quicktime.model")
            or None
        )
        # also check streams for tags
        for s in data.get("streams") or []:
            tags = s.get("tags") or {}
            if not creation:
                creation = tags.get("creation_time") or tags.get("time")
            if not device:
                device = tags.get("vendor") or tags.get("encoder")
        # normalize creation
        if creation:
            # ffprobe often returns ISO string
            try:
                d = datetime.fromisoformat(creation.replace("Z", "+00:00"))
                creation_iso = d.isoformat()
            except Exception:
                creation_iso = creation
        else:
            creation_iso = None
        return creation_iso, device
    except Exception:
        return None, None


def extract_date_from_filename(name: str) -> Optional[datetime]:
    """Try common filename date patterns."""
    patterns = [
        r"IMG[_\-](\d{8})",  # IMG-YYYYMMDD (check before generic)
        r"VID[_\-](\d{8})",  # VID-YYYYMMDD
        r"(\d{4})[_\-\.](\d{2})[_\-\.](\d{2})",  # YYYY-MM-DD variants
        r"^(\d{4})(\d{2})(\d{2})[_\-\.]",  # YYYYMMDD at start with separator
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 1:  # YYYYMMDD
                    date_str = groups[0]
                    year = int(date_str[:4])
                    # Sanity check: reject invalid years (1900-2100)
                    if year < 1900 or year > 2100:
                        continue
                    return datetime.strptime(date_str, "%Y%m%d")
                else:  # Y, M, D separate
                    year = int(groups[0])
                    if year < 1900 or year > 2100:
                        continue
                    return datetime(year, int(groups[1]), int(groups[2]))
            except:
                continue
    return None


def extract_date_and_device(p: Path) -> Tuple[datetime, Optional[str]]:
    date_iso, device = None, None
    if p.suffix.lower() in IMAGE_EXTS:
        date_iso, device = _get_exif_from_pillow(p)
    elif p.suffix.lower() in VIDEO_EXTS:
        date_iso, device = _get_video_metadata_with_ffprobe(p)

    if date_iso:
        try:
            dt = datetime.fromisoformat(date_iso)
        except:
            dt = None
    else:
        dt = None

    # Fallback chain: filename → mtime (if not 1970)
    if not dt:
        dt = extract_date_from_filename(p.name)
    if not dt:
        mtime_dt = datetime.fromtimestamp(p.stat().st_mtime)
        if mtime_dt.year != 1970:
            dt = mtime_dt
        else:
            dt = datetime.now()  # last resort

    dev_norm = sanitize_device_name(str(device)) if device else None
    return dt, dev_norm


# --- Inference of source if device missing ---
WHATSAPP_RE = re.compile(r"(WA\d{4}|-WA\d{4}|IMG-\d{8}-WA|VID-\d{8}-WA|WhatsApp)", re.I)
MESSENGER_RE = re.compile(r"(received_|facebook|fb_vid|messenger|video-)\w*", re.I)
INSTAGRAM_RE = re.compile(r"(insta|instagram|inshot|ig_)", re.I)
REDDIT_RE = re.compile(r"(reddit|r_)", re.I)
TELEGRAM_RE = re.compile(r"(Telegram|telegram|photo_|video_)", re.I)


def infer_source_from_name_and_path(p: Path) -> str:
    name = p.name
    parent = str(p.parent).lower()
    if WHATSAPP_RE.search(name) or "whatsapp" in parent:
        return "WhatsApp"
    if TELEGRAM_RE.search(name) or "telegram" in parent:
        return "Telegram"
    if INSTAGRAM_RE.search(name) or "instagram" in parent:
        return "Instagram"
    if REDDIT_RE.search(name) or "reddit" in parent:
        return "Reddit"
    if MESSENGER_RE.search(name) or "facebook" in parent or "messenger" in parent:
        return "Messenger"
    if name.startswith("PXL_"):
        return "Pixel"
    # downloads / browser common
    if (
        "download" in parent
        or "downloads" in parent
        or name.lower().startswith("image")
    ):
        return "Internet"
    return "Unknown"


def sanitize_device_name(dev: str) -> str:
    dev = dev.strip()
    # common transforms: "Apple iPhone 14 Pro" -> "iPhone 14 Pro"
    dev = re.sub(r"(?i)apple\s+", "", dev)
    dev = re.sub(r"(?i)samsung\s+", "", dev)
    dev = re.sub(r"[^0-9A-Za-z \-]", "", dev)
    dev = re.sub(r"\s+", " ", dev).strip()
    if not dev:
        return None
    return dev


# --- Naming utility ---
def build_target_name(
    dt: datetime, device_or_source: str, ext: str, directory: Path
) -> Path:
    # user wants DD.MM.YYYY DeviceName.ext
    date_part = dt.strftime("%d.%m.%Y")
    base = date_part
    if device_or_source:
        base = f"{base} {device_or_source}"
    candidate = directory / (base + ext)
    # iterate if exists
    i = 1
    while candidate.exists():
        candidate = directory / f"{base}_{i}{ext}"
        i += 1
    return candidate


def get_year_folder(base_dir: Path, dt: datetime) -> Path:
    """Return a folder path for the year extracted from dt (e.g., 2024/)."""
    # Check if current directory IS already a year folder
    try:
        year = int(base_dir.name)
        if 1900 <= year <= 2100:
            return base_dir
    except ValueError:
        pass

    year_dir = base_dir / str(dt.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir


# --- Optimization functions ---
def choose_image_cmd() -> Optional[str]:
    return MAGICK_BIN


def optimize_single_image(
    p: Path, no_rename: bool, conn: sqlite3.Connection
) -> Optional[Dict]:
    img_cmd = choose_image_cmd()
    if not img_cmd:
        return None
    # skip already-marked files or backups
    if "-original-optmz" in p.stem or safe_get_xattr(p, "status") == "optimized":
        return None

    # Check database
    if is_already_processed(conn, p):
        return None

    # gather metadata
    dt, device = extract_date_and_device(p)
    source = device or infer_source_from_name_and_path(p)
    dirpath = p.parent
    ext = p.suffix.lower()

    # plan names
    if no_rename:
        target_name = p
        year_dir = p.parent
    else:
        year_dir = get_year_folder(dirpath, dt)
        target_name = build_target_name(dt, source, ext, year_dir)
    backup_name = p.with_name(p.stem + "-original-optmz" + p.suffix)

    # temp path (to write optimized before replacing)
    temp_name = p.with_name(p.stem + ".optmz.tmp" + p.suffix)

    # run magick -> write temp
    try:
        # Preserve all metadata: remove -strip
        cmd = [
            img_cmd,
            str(p.resolve()),
            "-interlace",
            "Plane",
            "-quality",
            "85%",
            str(temp_name.resolve()),
        ]
        subprocess.run(
            cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not temp_name.exists():
            raise RuntimeError("ImageMagick failed (no temp output)")
        orig_size = p.stat().st_size
        new_size = temp_name.stat().st_size
        if new_size >= orig_size:
            temp_name.unlink()
            if not no_rename:
                p.rename(target_name)
                safe_set_xattr(target_name, "status", "no_improvement")
                safe_set_xattr(target_name, "optmz", datetime.now().isoformat())
                result = {
                    "original": str(p.resolve()),
                    "optimized": str(target_name.resolve()),
                    "original_size": orig_size,
                    "optimized_size": orig_size,
                    "space_saved": 0,
                    "status": "no_improvement",
                }
                save_optimization(conn, result)
                return result
            return None

        # backup original, move temp to target
        p.rename(backup_name)
        temp_name.rename(target_name)
        # tag metadata on target_name
        safe_set_xattr(target_name, "status", "optimized")
        safe_set_xattr(target_name, "original", str(backup_name.resolve()))
        safe_set_xattr(target_name, "optmz", datetime.now().isoformat())
        result = {
            "original": str(backup_name.resolve()),
            "optimized": str(target_name.resolve()),
            "original_size": orig_size,
            "optimized_size": new_size,
            "space_saved": orig_size - new_size,
            "status": "optimized",
        }
        save_optimization(conn, result)
        return result
    except Exception as e:
        if temp_name.exists():
            try:
                temp_name.unlink()
            except Exception:
                pass
        print(f"Error optimizing image {p}: {e}")
        return None


def optimize_single_video(
    p: Path, no_rename: bool, conn: sqlite3.Connection
) -> Optional[Dict]:
    if not FFMPEG_BIN:
        return None
    if "-original-optmz" in p.stem or safe_get_xattr(p, "status") == "optimized":
        return None

    # Check database
    if is_already_processed(conn, p):
        return None

    dt, device = extract_date_and_device(p)
    source = device or infer_source_from_name_and_path(p)
    dirpath = p.parent
    ext = p.suffix.lower()
    if no_rename:
        target_name = p
        year_dir = p.parent
    else:
        year_dir = get_year_folder(dirpath, dt)
        target_name = build_target_name(dt, source, ext, year_dir)
    backup_name = p.with_name(p.stem + "-original-optmz" + p.suffix)
    temp_name = p.with_name(p.stem + ".optmz.tmp" + p.suffix)

    try:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            str(p.resolve()),
            "-map_metadata",
            "0",  # preserve all metadata
            "-movflags",
            "+faststart",
            "-threads",
            str(max(1, os.cpu_count() or 1)),
            "-vcodec",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "28",
            "-acodec",
            "aac",
            "-b:a",
            "128k",
            str(temp_name.resolve()),
        ]
        subprocess.run(
            cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not temp_name.exists():
            raise RuntimeError("ffmpeg failed (no temp output)")
        orig_size = p.stat().st_size
        new_size = temp_name.stat().st_size
        if new_size >= orig_size:
            temp_name.unlink()
            if not no_rename:
                p.rename(target_name)
                safe_set_xattr(target_name, "status", "no_improvement")
                safe_set_xattr(target_name, "optmz", datetime.now().isoformat())
                result = {
                    "original": str(p.resolve()),
                    "optimized": str(target_name.resolve()),
                    "original_size": orig_size,
                    "optimized_size": orig_size,
                    "space_saved": 0,
                    "status": "no_improvement",
                }
                save_optimization(conn, result)
                return result
            return None

        # backup original, move temp to final target
        p.rename(backup_name)
        temp_name.rename(target_name)
        safe_set_xattr(target_name, "status", "optimized")
        safe_set_xattr(target_name, "original", str(backup_name.resolve()))
        safe_set_xattr(target_name, "optmz", datetime.now().isoformat())
        result = {
            "original": str(backup_name.resolve()),
            "optimized": str(target_name.resolve()),
            "original_size": orig_size,
            "optimized_size": new_size,
            "space_saved": orig_size - new_size,
            "status": "optimized",
        }
        save_optimization(conn, result)
        return result
    except Exception as e:
        if temp_name.exists():
            try:
                temp_name.unlink()
            except Exception:
                pass
        print(f"Error optimizing video {p}: {e}")
        return None


# wrapper
def optimize_worker(p: Path, no_rename: bool, db_path: str) -> Optional[Dict]:
    # Create per-thread connection
    conn = sqlite3.connect(db_path)
    try:
        if p.suffix.lower() in IMAGE_EXTS:
            result = optimize_single_image(p, no_rename, conn)
        elif p.suffix.lower() in VIDEO_EXTS:
            result = optimize_single_video(p, no_rename, conn)
        else:
            result = None
        return result
    except Exception as e:
        print(f"Unhandled error for {p}: {e}")
        return None
    finally:
        conn.close()


# --- Review / delete utilities ---
def preview_run(run_index: int, conn: sqlite3.Connection):
    runs = get_recent_runs(conn)
    if not runs:
        print("No runs recorded.")
        return
    if run_index < 0 or run_index >= len(runs):
        print(f"Invalid run index. There are {len(runs)} runs.")
        return

    run = runs[run_index]
    print(f"\n{'='*80}")
    print(f"Run from {run['timestamp']}")
    print(f"{'='*80}\n")

    # Print summary table
    print(
        f"{'Status':<10} {'Original Size':<15} {'New Size':<15} {'Saved':<12} {'File':<40}"
    )
    print("-" * 95)

    for file_data in run["files"]:
        orig = Path(file_data["original"])
        opt = Path(file_data["optimized"])
        status = "✓ OK" if file_data["space_saved"] > 0 else "- SKIP"

        orig_size_str = human_readable(file_data.get("original_size", 0))
        new_size_str = human_readable(file_data.get("optimized_size", 0))
        saved_str = human_readable(file_data.get("space_saved", 0))

        # Show optimized filename (truncate if too long)
        display_name = opt.name if len(opt.name) <= 40 else opt.name[:37] + "..."

        print(
            f"{status:<10} {orig_size_str:<15} {new_size_str:<15} {saved_str:<12} {display_name:<40}"
        )

    print("-" * 95)
    total_saved = human_readable(run.get("total_space_saved", 0))
    print(f"{'TOTAL':<10} {'':15} {'':15} {total_saved:<12}\n")


def find_renamed_file(original_path: Path) -> Optional[Path]:
    """Try to find a file that has been renamed with common suffixes."""
    if original_path.exists():
        return original_path

    # Check common rename patterns in the same directory
    parent = original_path.parent
    stem = original_path.stem
    suffix = original_path.suffix

    # Common patterns: -optmzd-fail, -failed, _backup, etc.
    patterns = [
        f"{stem}-optmzd-fail{suffix}",
    ]

    for pattern in patterns:
        candidate = parent / pattern
        if candidate.exists():
            return candidate

    # Try to find by matching stem in same directory
    try:
        for file in parent.glob(f"{stem}*{suffix}"):
            if file.name != original_path.name:
                return file
    except:
        pass

    return None


def delete_originals_from_db(conn: sqlite3.Connection):
    cursor = conn.execute("SELECT id, original_path FROM optimizations")
    rows = cursor.fetchall()

    if not rows:
        print("No original paths found in database.")
        return

    print(f"Found {len(rows)} originals in database.")

    moved = 0
    found_renamed = 0
    not_found = 0

    for row_id, original_path_str in rows:
        p = Path(original_path_str)

        # Try to find the file (even if renamed)
        actual_file = find_renamed_file(p)

        if not actual_file:
            not_found += 1
            continue

        if actual_file != p:
            found_renamed += 1
            print(f"Found renamed: {p.name} → {actual_file.name}")
            # Update database with new path
            conn.execute(
                "UPDATE optimizations SET original_path = ? WHERE id = ?",
                (str(actual_file.resolve()), row_id),
            )
            conn.commit()

        if not is_safe_to_delete(actual_file):
            print(f"🚫 Skipping unsafe path: {actual_file}")
            continue
        else:
            try:
                actual_file.unlink()
                moved += 1
            except Exception as e:
                print(f"Failed to delete {actual_file}: {e}")

    print(f"\nSummary:")
    if found_renamed > 0:
        print(f"  Found {found_renamed} renamed files")
    else:
        print(f"  Deleted {moved}/{len(rows)} files")
    if not_found > 0:
        print(f"  Could not find {not_found} files")


SAFE_DELETE_ROOTS = {
    "/",
    "/home",
    "/root",
    "/etc",
    "/usr",
    "/var",
    "/opt",
    "/mnt",
    "/media",
    "/tmp",
}


def is_safe_to_delete(p: Path) -> bool:
    """Check that a path is safe to delete/move."""
    if not p.exists():
        return False
    if str(p.resolve()) in SAFE_DELETE_ROOTS:
        return False
    if p.is_dir():
        return False
    if len(p.parts) <= 2:  # shallow paths like /x
        return False
    return True


def fix_paths(conn: sqlite3.Connection, old_prefix: str, new_prefix: str):
    """Replace old path prefix with new one for all paths in the DB."""
    cursor = conn.cursor()
    for column in ("original_path", "optimized_path"):
        cursor.execute(
            f"""
            UPDATE optimizations
            SET {column} = REPLACE({column}, ?, ?)
            WHERE {column} LIKE ?;
            """,
            (old_prefix, new_prefix, f"{old_prefix}%"),
        )
    conn.commit()


def detect_and_offer_fix(conn: sqlite3.Connection):
    """Detect if the media folder has moved and automatically fix DB paths."""
    cur = conn.cursor()
    cur.execute("SELECT original_path, optimized_path FROM optimizations LIMIT 200")
    rows = cur.fetchall()
    if not rows:
        return

    originals = [Path(r[0]) for r in rows if r[0]]
    optimized = [Path(r[1]) for r in rows if r[1]]

    missing = [p for p in originals + optimized if not p.exists()]
    if len(missing) < len(originals + optimized) * 0.7:
        return  # not enough missing to assume a move

    # Find the common old prefix (the old base folder)
    old_prefix = os.path.commonpath([str(p) for p in missing])
    base_name = Path(old_prefix).name
    home = Path.home()

    print(f"\n🔍 Most files missing. Looking for moved folder matching: {base_name}")

    # Search a few candidate locations under home
    candidates = [c for c in home.rglob(base_name) if c.is_dir()]
    if not candidates:
        print(
            "❌ Could not find a folder with the same name under your home directory."
        )
        return

    # Try to validate each candidate by filename overlap
    for candidate in candidates:
        test_files = missing[:10]
        matches = sum((candidate / p.name).exists() for p in test_files)
        if matches >= len(test_files) * 0.6:
            new_prefix = str(candidate)
            print(
                f"\n✅ Found likely new location:\nOld prefix: {old_prefix}\nNew prefix: {new_prefix}"
            )
            choice = (
                input("Update all paths in database automatically? [y/N]: ")
                .strip()
                .lower()
            )
            if choice == "y":
                fix_paths(conn, old_prefix, new_prefix)
                print("🛠️  Database paths updated successfully.\n")
            return

    print("❌ No valid new prefix found that matches your files.")


def audit_missing_files(conn: sqlite3.Connection):
    cursor = conn.execute("SELECT id, original_path, optimized_path FROM optimizations")
    rows = cursor.fetchall()

    # Only check for cases where BOTH files are missing
    missing = []
    for r in rows:
        orig_exists = Path(r[1]).exists()
        opt_exists = Path(r[2]).exists()

        # Only flag if both are missing (actual problem)
        if not orig_exists and not opt_exists:
            missing.append((r[0], r[1], r[2]))

    if not missing:
        print("✅ All files exist for recorded optimizations.")
        return

    print(f"⚠️  {len(missing)} entries found where both files are missing.\n")

    for i, (id, orig, opt) in enumerate(missing[:10], 1):
        print(f"  ID: {id}\n  Original: {orig}\n  Optimized: {opt}\n")

    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more\n")

    choice = (
        input(
            "It looks like these files are missing.\nDid you (d)elete them, (m)ove the folder, or (s)kip? [d/m/s]: "
        )
        .strip()
        .lower()
    )

    if choice == "s":
        print("➡️  Skipped audit fixes.")
        return

    elif choice == "d":
        confirm = (
            input(f"Delete {len(missing)} missing entries from DB? [y/N]: ")
            .strip()
            .lower()
        )
        if confirm == "y":
            ids = [str(r[0]) for r in missing]
            conn.execute(f"DELETE FROM optimizations WHERE id IN ({','.join(ids)})")
            conn.commit()
            print(f"🗑️  Deleted {len(ids)} missing entries.")
        return

    elif choice != "m":
        print("❌ Invalid choice.")
        return

    sample_paths = [Path(m[1]) for m in missing[:100]]
    if not sample_paths:
        print("❌ No original paths available for folder detection.")
        return

    old_root = os.path.commonpath([str(p.parent) for p in sample_paths])
    cwd = Path.cwd()
    sample_filenames = [p.name for p in sample_paths[:50]]

    print(f"\n🔍 Detecting moved folder. Old location was: {old_root}")

    found_new_prefix = None

    if (cwd / "2023").is_dir() or (cwd / "2024").is_dir():
        print("✅ Current directory contains year folders, checking files...")
        matches = sum(1 for fn in sample_filenames if any(cwd.rglob(fn)))
        match_rate = matches / len(sample_filenames)
        print(f"File match rate: {int(match_rate*100)}%")
        if match_rate >= 0.1:
            found_new_prefix = str(cwd)
        else:
            print(
                f"⚠️  Low match rate, but year folders exist. Assuming correct location."
            )
            found_new_prefix = str(cwd)

    if not found_new_prefix:
        print("❌ Could not automatically detect the new folder.")
        print(f"Current directory: {cwd}")
        manual = input(
            "Enter new base path manually (or leave blank to use current dir): "
        ).strip()
        found_new_prefix = manual if manual else str(cwd)

    # Confirm and replace paths
    confirm = (
        input(
            f"\nFound possible new folder:\n  {found_new_prefix}\nReplace all occurrences of:\n  {old_root}\n→ {found_new_prefix} ? [y/N]: "
        )
        .strip()
        .lower()
    )

    if confirm != "y":
        print("➡️  Skipped path replacement.")
        return

    conn.execute(
        """
        UPDATE optimizations
        SET original_path = REPLACE(original_path, ?, ?),
            optimized_path = REPLACE(optimized_path, ?, ?)
        WHERE original_path LIKE ? OR optimized_path LIKE ?
        """,
        (
            old_root,
            found_new_prefix,
            old_root,
            found_new_prefix,
            f"{old_root}%",
            f"{old_root}%",
        ),
    )
    conn.commit()

    print(f'✅ Updated paths from "{old_root}" → "{found_new_prefix}".')
    print("🔁 Re-running audit to confirm...\n")
    audit_missing_files(conn)


# --- Main orchestration ---
def collect_files(base: Path) -> List[Path]:
    files = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            files.append(p)
    return files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="optmz - optimize images/videos, preserve metadata, smart rename"
    )
    parser.add_argument(
        "-r",
        "--review",
        type=int,
        nargs="?",
        const=0,
        help="Review previous run index (0 = last).",
    )
    parser.add_argument(
        "-nr",
        "--no-rename",
        action="store_true",
        help="Do not rename optimized files; keep original filenames.",
    )

    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Delete all original backup files directly from database paths (non-interactive).",
    )

    parser.add_argument(
        "-a",
        "--audit",
        action="store_true",
        help="Check for database entries where both original and optimized files are missing.",
    )

    parser.add_argument(
        "-fp",
        "--fix-paths",
        nargs=2,
        metavar=("TARGET", "NEW_PREFIX"),
        help="Fix stored paths in DB after moving folders. TARGET = originals|optimized",
    )

    args = parser.parse_args()

    # Initialize database
    conn = init_db()

    if args.clean:
        delete_originals_from_db(conn)
        conn.close()
        return

    if args.review is not None:
        preview_run(args.review, conn)
        conn.close()
        return

    if args.audit:
        audit_missing_files(conn)
        conn.close()
        return

    if args.fix_paths:
        target, new_prefix = args.fix_paths
        fix_paths(conn, target, new_prefix)
        conn.close()
        return

    base = Path(".").resolve()
    # Around line 1030, after "files = collect_files(base)"
    files = collect_files(base)
    if not files:
        print("No media files found to optimize.")
        conn.close()
        return

    # Filter out already processed files
    files_to_process = []
    for f in files:
        if is_already_processed(conn, f):
            continue
        if f.suffix.lower() in IMAGE_EXTS and not MAGICK_BIN:
            continue
        if f.suffix.lower() in VIDEO_EXTS and not FFMPEG_BIN:
            continue
        files_to_process.append(f)

    if not files_to_process:
        print("No new files to optimize (all already processed).")
        conn.close()
        return

    images = [f for f in files_to_process if f.suffix.lower() in IMAGE_EXTS]
    videos = [f for f in files_to_process if f.suffix.lower() in VIDEO_EXTS]

    print(f"\nFound {len(files_to_process)} files to optimize:")
    print(f"  Images: {len(images)}")
    print(f"  Videos: {len(videos)}")

    # Show files in compact table format
    if len(files_to_process) <= 50:
        print("\nFiles to process:")
        for f in files_to_process:
            ftype = "IMG" if f.suffix.lower() in IMAGE_EXTS else "VID"
            print(f"  [{ftype}] {f.name}")
    else:
        print(f"\nShowing first 25 and last 25 of {len(files_to_process)} files:")
        for f in files_to_process[:25]:
            ftype = "IMG" if f.suffix.lower() in IMAGE_EXTS else "VID"
            print(f"  [{ftype}] {f.name}")
        print(f"  ... ({len(files_to_process) - 50} more files) ...")
        for f in files_to_process[-25:]:
            ftype = "IMG" if f.suffix.lower() in IMAGE_EXTS else "VID"
            print(f"  [{ftype}] {f.name}")

    choice = input("\nProceed with optimization? [y/N]: ").strip().lower()
    if choice != "y":
        print("Cancelled.")
        conn.close()
        return

    print(f"Starting optimization with {MAX_WORKERS} workers...")
    files = files_to_process  # Replace with filtered list

    entries = []
    interrupted = False
    use_tqdm = tqdm is not None
    pbar = tqdm(total=len(files), desc="Optimizing", unit="file") if use_tqdm else None

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(optimize_worker, p, args.no_rename, OPT_DB_FILE): p
                for p in files
            }
            for fut in as_completed(futures):
                res = None
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"Worker exception: {e}")
                if res:
                    entries.append(res)
                if pbar:
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user (Ctrl+C)")
        interrupted = True
    finally:
        if pbar:
            pbar.close()

    if entries:
        total = sum(e.get("space_saved", 0) for e in entries)
        status_msg = "⚠️  Partial run" if interrupted else "Run complete"
        print(f"\n{status_msg}. Files optimized: {len(entries)}")
        print(
            f"Total space saved (if you delete original backups): {human_readable(total)}"
        )
    else:
        print("No files were optimized.")

    conn.close()


if __name__ == "__main__":
    # Minor sanity check for required binaries
    if not MAGICK_BIN:
        print(
            "Warning: ImageMagick (magick/convert) not found. Images will be skipped."
        )
    if not FFMPEG_BIN:
        print("Warning: ffmpeg not found. Videos will be skipped.")

    main()  # KeyboardInterrupt is now handled inside main()
