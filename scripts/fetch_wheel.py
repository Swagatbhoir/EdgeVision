import urllib.request
import re
import sys

INDEX = "https://pypi.org/simple/opencv-python/"
TARGET = "opencv_python-4.11.0.86"

def main():
    req = urllib.request.Request(INDEX, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")

    # Find the cp311/abi3 win_amd64 wheel for the target version
    hrefs = re.findall(r'href="([^"]+\.whl[^"]*)"', html)
    cands = [h for h in hrefs if TARGET in h and ("cp37-abi3" in h or "cp311" in h) and "win_amd64" in h]
    if not cands:
        cands = [h for h in hrefs if TARGET in h and "win_amd64" in h]
    if not cands:
        print("NO MATCH for", TARGET, file=sys.stderr)
        return 1
    url = cands[0]
    if url.startswith("//"):
        url = "https:" + url
    fname = url.split("/")[-1]
    print("Downloading", url, flush=True)

    def dot_reader(block):
        total = 0
        for chunk in iter(lambda: block.read(1024 * 512), b""):
            yield chunk
            total += len(chunk)
            if total % (5 * 1024 * 1024) < 1024 * 512:
                print(f"  ...{total // (1024*1024)} MB", flush=True)

    req2 = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req2, timeout=120) as src, open(fname, "wb") as dst:
        for chunk in dot_reader(src):
            dst.write(chunk)
    print("Saved", fname, flush=True)
    return 0

sys.exit(main())