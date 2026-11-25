#!/usr/bin/env python3
"""Selective iSign video extraction. Requires requests and authorized HF access."""
import argparse
import bisect
import csv
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
import zlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests

BASE = 'https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/'
PARTS = ['iSign-videos_v1.1_part_aa', 'iSign-videos_v1.1_part_ab']
CSV_BASE = 'https://raw.githubusercontent.com/Analroy/SpaMo-OF/main/Data/'
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}
RANGE = re.compile(r'bytes (\d+)-(\d+)/(\d+)')


def fetch_range(session, url, start, end):
    for attempt in range(4):
        try:
            with session.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 90)) as r:
                if r.status_code in (401, 403):
                    raise PermissionError('Hugging Face denied access. Accept iSign conditions and use an authorized read token.')
                if r.status_code == 200:
                    raise RuntimeError('Server ignored Range request. Aborting rather than downloading the entire archive.')
                r.raise_for_status()
                m = RANGE.fullmatch(r.headers.get('Content-Range', ''))
                if r.status_code != 206 or not m or tuple(map(int, m.groups()[:2])) != (start, end):
                    raise RuntimeError('Server did not return the exact requested byte range.')
                data = r.content
                if len(data) != end - start + 1:
                    raise IOError('Incomplete byte range')
                return data, int(m.group(3))
        except (requests.RequestException, IOError) as e:
            if isinstance(e, PermissionError) or attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RemoteZip(io.RawIOBase):
    """Seekable virtual concatenation of raw ZIP parts; no full-file download."""
    def __init__(self, urls, sizes, token):
        super().__init__()
        self.urls = urls
        self.ends = []
        for size in sizes:
            self.ends.append(size + (self.ends[-1] if self.ends else 0))
        self.session = requests.Session()
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'
        self.pos = 0
        self.cache_start = 0
        self.cache = b''

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else (self.pos if whence == 1 else self.ends[-1]) + offset
        if pos < 0: raise ValueError('Negative seek')
        self.pos = pos
        return pos

    def _read_range(self, start, length):
        chunks = []
        while length:
            i = bisect.bisect_right(self.ends, start)
            base = self.ends[i-1] if i else 0
            n = min(length, self.ends[i]-start, 8*1024*1024)
            data, size = fetch_range(self.session, self.urls[i], start-base, start-base+n-1)
            if size != self.ends[i]-base: raise RuntimeError('Remote archive size changed')
            chunks.append(data)
            start += n
            length -= n
        return b''.join(chunks)

    def read(self, size=-1):
        remaining = max(0, self.ends[-1]-self.pos)
        n = remaining if size is None or size < 0 else min(size, remaining)
        if not n: return b''
        start = self.pos
        offset = start-self.cache_start
        if offset >= 0 and offset+n <= len(self.cache):
            data = self.cache[offset:offset+n]
        elif n < 65536:
            self.cache_start = start
            self.cache = self._read_range(start, min(65536, remaining))
            data = self.cache[:n]
        else:
            data = self._read_range(start, n)
        self.pos += len(data)
        return data

    def close(self):
        self.session.close()
        super().close()


def probe(urls, token):
    with requests.Session() as session:
        if token: session.headers['Authorization'] = f'Bearer {token}'
        sizes = [fetch_range(session, url, 0, 0)[1] for url in urls]
    return sizes


def read_splits(out):
    rows = []
    for split, filename in [('train','train.csv'),('val','dev.csv'),('test','test.csv')]:
        response = requests.get(CSV_BASE+filename, timeout=60)
        response.raise_for_status()
        data = response.content
        (out/'splits'/filename).write_bytes(data)
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
        if not {'video_name','text'}.issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected CSV header in {filename}: {reader.fieldnames}')
        count = 0
        for row in reader:
            if None in row or not row.get('video_name') or not row.get('text'):
                raise ValueError(f'Malformed row in {filename}: {reader.line_num}')
            rows.append({'id':row['video_name'].strip(),'text':row['text'],'split':split})
            count += 1
        print(f'{split}: {count} rows', flush=True)
    return rows


def get_token():
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            pass
    if not token:
        try:
            from huggingface_hub import get_token as hf_get_token
            token = hf_get_token()
        except ImportError:
            pass
    if not token:
        raise SystemExit('HF_TOKEN is required. Accept access at https://huggingface.co/datasets/Exploration-Lab/iSign and add a read token to Kaggle Secrets or log in with hf auth login.')
    return token


def verified(path, info):
    if not path.is_file() or path.stat().st_size != info.file_size: return False
    crc = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): crc = zlib.crc32(block, crc)
    return crc & 0xffffffff == info.CRC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='isign_spamo_10k')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='Download only first N matched unique videos for testing; default all')
    p.add_argument('--index-only', action='store_true')
    a = p.parse_args()
    out = Path(a.output).resolve()
    (out/'splits').mkdir(parents=True, exist_ok=True)
    (out/'videos').mkdir(exist_ok=True)
    rows = read_splits(out)
    token = get_token()
    urls = [BASE+x for x in PARTS]
    sizes = probe(urls, token)
    print('Remote archive size (GB):', round(sum(sizes)/1e9,3), flush=True)
    by_id = defaultdict(list)
    with RemoteZip(urls,sizes,token) as remote, zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if not info.is_dir() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_id[path.stem].append(info)
    matches = {}
    missing, ambiguous = [], []
    for row in rows:
        uid = row['id']
        key = PurePosixPath(uid).name
        if PurePosixPath(key).suffix.lower() in VIDEO_SUFFIXES: key = PurePosixPath(key).stem
        candidates = by_id.get(key, [])
        if not candidates: missing.append(row); continue
        if len(candidates) != 1: ambiguous.append(row); continue
        info = candidates[0]
        name = PurePosixPath(info.filename).name
        row['video_file'] = 'videos/'+name
        matches[info.filename] = (info,out/'videos'/name)
    report = {'split_counts':dict(Counter(r['split'] for r in rows)), 'rows':len(rows), 'unique_videos':len(matches), 'missing':missing, 'ambiguous':ambiguous, 'compressed_bytes':sum(i.compress_size for i,_ in matches.values()), 'extracted_bytes':sum(i.file_size for i,_ in matches.values())}
    (out/'match_report.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k not in ('missing','ambiguous')},flush=True)
    print('Missing:',len(missing),'Ambiguous:',len(ambiguous),flush=True)
    if missing or ambiguous: raise SystemExit('Exact matching incomplete; inspect match_report.json. No video downloads started.')
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','text','split','video_file']); w.writeheader(); w.writerows(rows)
    if a.index_only: return
    tasks = sorted(matches.values(), key=lambda x:x[0].header_offset)
    if a.limit: tasks=tasks[:a.limit]
    local = threading.local()
    def download(task):
        info,path=task
        if verified(path,info): return 'existing'
        if not hasattr(local,'archive'):
            local.remote=RemoteZip(urls,sizes,token)
            local.archive=zipfile.ZipFile(local.remote)
        tmp=path.with_suffix(path.suffix+'.partial')
        try:
            with local.archive.open(info) as src, tmp.open('wb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            if not verified(tmp,info): raise IOError('CRC or size mismatch')
            tmp.replace(path)
            return 'downloaded'
        finally:
            tmp.unlink(missing_ok=True)
    counts=Counter(); failures=[]
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as pool:
        futures={pool.submit(download,t):t[0].filename for t in tasks}
        for n,future in enumerate(as_completed(futures),1):
            try: counts[future.result()]+=1
            except Exception as e: failures.append({'file':futures[future],'error':str(e)})
            if n%100==0 or n==len(tasks): print(f'{n}/{len(tasks)}: {dict(counts)}, failed={len(failures)}',flush=True)
    (out/'download_report.json').write_text(json.dumps({'counts':dict(counts),'failures':failures,'limited_run':bool(a.limit)},indent=2))
    if failures: raise SystemExit('Some videos failed. Rerun to retry; verified files will be skipped.')
    print('Selected download complete.' if not a.limit else 'Limited test complete.',out,flush=True)


if __name__=='__main__': main()
