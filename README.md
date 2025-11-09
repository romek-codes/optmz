# 🧠 optmz — Smart Media Optimizer

A fast, metadata-preserving media optimizer for images and videos with intelligent organization and crash-safe processing.

---

## ✨ Highlights

* ⚙️ **Non-destructive** — originals renamed with `-original-optmz` suffix
* 🧠 **Metadata-safe** — preserves EXIF, camera model, timestamps
* 💾 **Saves space** — tracks every run in SQLite database
* 🗂️ **Auto-organizes** — files sorted into year folders (e.g., `2024/22.09.2024 iPhone 14 Pro.jpg`)
* 🚀 **Parallel optimization** using 4 worker threads
* 🔍 **Review mode** with clean summary tables and backup management
* 🧩 **Smart source detection** — infers WhatsApp / Messenger / Instagram / Telegram / Reddit from filenames
* 💥 **Crash-resistant** — SQLite stores progress per-file, safe to Ctrl+C and resume
* 📅 **Intelligent date fallback** — EXIF → filename patterns → mtime (skips Unix epoch timestamps)
* 🔎 **Audit mode** — detects moved folders and fixes database paths automatically

---

## 🧱 Installation (Nix Flakes)

### 1️⃣ Add to your flake inputs

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    optmz.url = "github:romek-codes/optmz";
  };
}
```

### 2️⃣ Use via `nix run`

```bash
nix run github:romek-codes/optmz
```

Or if already declared as a flake input:

```bash
nix run .#optmz
```

---

### 3️⃣ (Optional) Add to your system / Home Manager

If you want `optmz` always available on your system:

#### NixOS system configuration

```nix
environment.systemPackages = with pkgs; [
  inputs.optmz.packages.${system}.default
];
```

#### Home Manager configuration

```nix
home.packages = [
  inputs.optmz.packages.${system}.default
];
```

---

## ⚡ Development Shell (Nix)

To hack on `optmz` locally:

```bash
nix develop
```

or, with legacy Nix:

```bash
nix-shell
```

The shell includes:

* 🐍 Python (with Pillow + tqdm)
* 🪄 ImageMagick
* 🎞️ FFmpeg
* 🖼️ Loupe (image preview)
* 📺 VLC (video preview)

---

## 🔧 Manual Installation (non-Nix systems)

You can also install manually:

```bash
git clone https://github.com/romek-codes/optmz
cd optmz
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
python optmz.py
```

---

## 🧩 Usage

### Optimize all media in current directory

```bash
optmz
```

Shows a preview of files to process and asks for confirmation before proceeding.

Files are:
- Optimized with ImageMagick (images) or FFmpeg (videos)
- Renamed to `DD.MM.YYYY DeviceName.ext` format
- Moved to year folders (e.g., `2024/22.09.2024 iPhone 14 Pro.jpg`)
- Original backups kept as `filename-original-optmz.ext`

### Keep original names (no renaming/moving)

```bash
optmz --no-rename
```

Files stay in place, only optimization is applied.

### Review last run and backup originals

```bash
optmz -r
```

Shows a clean table:
```
Status     Original Size   New Size        Saved        File
-----------------------------------------------------------------------------------------------
✓ OK       1.08 MB         725.02 KB       379.29 KB    22.09.2024 iPhone 14 Pro.jpg
- SKIP     284.27 KB       284.27 KB       0.00 B       19.11.2024 WhatsApp.jpg
-----------------------------------------------------------------------------------------------
TOTAL                                      2.06 MB
```

Then automatically moves all backup files (`-original-optmz` copies) to `optmz-backup/` folder, updating database paths accordingly.

### Review older runs

```bash
optmz -r 1  # Review run before last
optmz -r 2  # Two runs ago
```

### Clean all backups from database

```bash
optmz --clean
```

Moves all original backup files tracked in the database to `optmz-backup/` folder. Automatically detects renamed files (e.g., `IMG-20181224-WA0042-optmzd-fail.jpg`).

### Audit database and fix paths

```bash
optmz --audit
```

Checks for missing files in database and offers to:
- Delete entries where both original and optimized files are missing
- Auto-detect moved folders and update all paths in database
- Handle renamed backups automatically

Great for when you've moved your media folder to a new location!

---

## 🗂️ File Organization

### Default behavior (with renaming):

```
Before:
  IMG_20240922_143052.jpg
  WA0042.jpg
  PXL_20231215_091234.mp4

After:
  2024/
    22.09.2024 Pixel 6.jpg
    15.12.2023 Pixel 6.mp4
  2024/
    22.09.2024 WhatsApp.jpg
  
  IMG_20240922_143052-original-optmz.jpg  # backup
  WA0042-original-optmz.jpg                # backup
  PXL_20231215_091234-original-optmz.mp4  # backup
```

### Smart folder detection

If files are already in a year folder (e.g., `2024/`), they won't be moved to nested year folders. The script detects this automatically.

### Database tracking

All operations saved to `.optmzd.db`:
```sql
SELECT * FROM optimizations;
-- Shows: timestamp, paths, sizes, space saved, status
```

---

## 🧩 Smart Features

### Source Detection
Automatically identifies file sources:
- **WhatsApp**: `WA####`, `IMG-########-WA`
- **Telegram**: `photo_`, `video_`, folder name
- **Instagram**: `insta`, `ig_`, folder name  
- **Messenger**: `received_`, `fb_vid`
- **Reddit**: `reddit`, `r_`
- **Pixel phones**: `PXL_` prefix
- **Internet**: Downloads folder or generic names

### Date Extraction Priority
1. **EXIF metadata** (DateTimeOriginal for images)
2. **Video metadata** (creation_time via FFprobe)
3. **Filename patterns** (YYYYMMDD, IMG-20240922, etc.)
4. **File modification time** (if not Unix epoch)
5. **Current date** (last resort)

### Crash Safety
- Each file commits to SQLite immediately after processing
- Interrupted runs can resume — already-processed files are skipped via:
  - Database path checking
  - Extended attributes (xattr) status checking (`optimized` or `no_improvement`)
  - Filename pattern matching (DD.MM.YYYY format)
  - Backup file existence

### Skip Logic
Files are automatically skipped if:
- Already in database (by path)
- Have xattr status of `optimized` or `no_improvement`
- Filename matches optimized pattern (DD.MM.YYYY)
- Backup file with `-original-optmz` suffix exists

---

## 🧩 Nix Details

The flake exports:

```nix
packages.${system}.default # the main CLI binary: optmz
devShells.${system}.default # development shell
```

So if you're using it in another flake:

```nix
{
  inputs.optmz.url = "github:romek-codes/optmz";
  outputs = { self, nixpkgs, optmz, ... }: {
    packages.${system}.default = optmz.packages.${system}.default;
  };
}
```

---

## 🧠 Philosophy

* 🧍 **Human-safe**: never overwrites originals, always creates backups
* 💥 **Crash-resistant**: SQLite ensures no lost work, safe to interrupt anytime
* 🧩 **Nix-first**: designed for reproducibility and declarative workflows
* 🪶 **Metadata-aware**: filenames automatically derived from EXIF (e.g., `26.11.2001 iPhone 14 Pro.jpg`)
* ⚡ **Performance-conscious**: uses 4 worker threads, leaving system responsive
* 🔍 **Smart recovery**: automatically detects moved folders and renamed backups

---

## 📊 Optimization Settings

### Images (ImageMagick)
- Quality: 85%
- Preserves all metadata (no stripping)
- Progressive JPEG encoding (interlaced)

### Videos (FFmpeg)
- Codec: H.264 (libx264)
- CRF: 28 (good quality/size balance)
- Audio: AAC 128kbps
- Preserves all metadata streams
- Faststart flag for web streaming

---

## 🛠️ Command Reference

```bash
# Basic optimization
optmz                    # Optimize with confirmation prompt
optmz --no-rename        # Optimize without renaming/organizing

# Review and cleanup
optmz -r                 # Review last run and backup originals
optmz -r 1               # Review previous run
optmz --clean            # Delete all original files

# Database maintenance
optmz --audit            # Check for missing files, auto-fix moved folders
optmz --fix-paths OLD NEW  # Manually fix paths in database
```

---

## ⚖️ License

MIT © 2025 Romek  
Pull requests and issues welcome 💚
