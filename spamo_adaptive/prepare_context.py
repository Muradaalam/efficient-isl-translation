"""Optional Google Cloud Translation v2 reference bank, using only training texts.
Requires enabled billing/API and a restricted key. No video files are sent.
"""
import argparse
import csv
import html
import os
from pathlib import Path
import requests
from extract_features import read_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true',help='Actually call the billable API')
    p.add_argument('--limit',type=int,default=0,help='Optional smaller reference bank (deviation from full training pool)')
    a=p.parse_args()
    rows=[r for r in read_manifest(a.data) if r['split']=='train']
    if a.limit: rows=rows[:a.limit]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if out.exists():
        with out.open(newline='',encoding='utf-8') as f: done={r['id']:r for r in csv.DictReader(f)}
    rows=[r for r in rows if r['id'] not in done]
    print('Pending references:',len(rows),'Estimated input characters across 3 languages:',3*sum(len(r['text']) for r in rows))
    if not a.execute: return
    key=os.environ.get('GOOGLE_TRANSLATE_KEY')
    if not key:
        from kaggle_secrets import UserSecretsClient
        key=UserSecretsClient().get_secret('GOOGLE_TRANSLATE_KEY')
    fields=['id','hi','fr','es']
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]; results=[{'id':r['id']} for r in batch]
        for lang in fields[1:]:
            response=requests.post('https://translation.googleapis.com/language/translate/v2',headers={'X-Goog-Api-Key':key},json={'q':[r['text'] for r in batch],'source':'en','target':lang,'format':'text'},timeout=60)
            if not response.ok: raise RuntimeError(f'Google Translate HTTP {response.status_code}; check API configuration and billing')
            translations=response.json()['data']['translations']
            if len(translations)!=len(batch): raise ValueError('Unexpected response length')
            for row,tr in zip(results,translations): row[lang]=html.unescape(tr['translatedText'])
        existing=out.exists() and out.stat().st_size>0
        with out.open('a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if not existing: w.writeheader()
            w.writerows(results)
        print(f'{min(start+32,len(rows))}/{len(rows)} saved',flush=True)

if __name__=='__main__': main()
