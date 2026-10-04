from pathlib import Path
import zipfile
import gdown

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

FILES = {
    "ur5e.zip": "1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc",
    "robotiq_2f_85.zip": "1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX",
    "bowl.zip": "1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM",
}

for filename, file_id in FILES.items():
    archive = ROOT / filename
    if not archive.exists():
        print(f"Downloading {filename}...")
        gdown.download(id=file_id, output=str(archive), quiet=False)
    else:
        print(f"Already downloaded: {filename}")

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(ASSETS)
        print(f"Extracted {filename}")

print("\nAsset check:")
for path in [
    ASSETS / "ur5e" / "ur5e.urdf",
    ASSETS / "robotiq_2f_85" / "robotiq_2f_85.urdf",
    ASSETS / "bowl" / "bowl.urdf",
]:
    print(path, "OK" if path.exists() else "MISSING")
