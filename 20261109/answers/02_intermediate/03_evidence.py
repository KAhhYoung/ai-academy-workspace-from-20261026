from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cv2
base = Path(__file__).resolve().parent
data = base / 'data'
out = base / 'outputs'
out.mkdir(exist_ok=True)
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

doc = lesson['documents']['doc01']
words = doc['ocr']['tokens']
fields = {}
for key, label in [('issued_at','DATE:'), ('supplier','SUPPLIER:'), ('total_amount','TOTAL:')]:
    anchor = next(w for w in words if w['text'] == label)
    # 같은 block/paragraph/line이고 라벨 오른쪽에 있는 단어만 선택합니다.
    # 해설 A03
        # OCR로 인식한 단어 목록(words)에서 특정 기준 단어(anchor)의
    # 오른쪽에 위치한 단어들만 추출합니다.
    #
    # words  : OCR로 인식한 전체 단어 정보 목록
    # anchor : 위치를 판단할 기준 단어 (예: "DATE:", "TOTAL:")
    #
    # ① w['block'] == anchor['block']
    #    → 기준 단어와 동일한 텍스트 블록에 속하는지 확인
    #
    # ② w['paragraph'] == anchor['paragraph']
    #    → 기준 단어와 동일한 문단에 속하는지 확인
    #
    # ③ w['line'] == anchor['line']
    #    → 기준 단어와 동일한 줄에 속하는지 확인
    #
    # ④ w['bbox'][0] >= anchor['bbox'][2]
    #    → 현재 단어의 왼쪽 X좌표가 기준 단어의 오른쪽 X좌표 이상인지 확인
    #    → 즉, 기준 단어의 오른쪽에 있는 단어만 선택
    #
    # bbox = [x1, y1, x2, y2]
    # x1: 왼쪽 좌표, x2: 오른쪽 좌표
    #
    # 최종 결과:
    # 기준 단어와 같은 블록·문단·줄에 있으면서,
    # 기준 단어의 오른쪽에 있는 단어들을 selected 리스트에 저장

    selected = [
        w for w in words
        if (
            w['block'],
            w['paragraph'],
            w['line']
        ) == (
            anchor['block'],
            anchor['paragraph'],
            anchor['line']
        )
        and w['bbox'][0] >= anchor['bbox'][2]
    ]
# 해설 끝 A03
    raw = ' '.join(w['text'] for w in selected)
    value = int(raw) if key == 'total_amount' else raw
    box = [min(w['bbox'][0] for w in selected),min(w['bbox'][1] for w in selected),max(w['bbox'][2] for w in selected),max(w['bbox'][3] for w in selected)]
    fields[key] = {'raw': raw, 'value': value, 'confidence': min(w['conf'] for w in selected), 'bbox': box, 'coordinate_space':'original_pixels'}
(out / 'p03_evidence.json').write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding='utf-8')
print({k:v['value'] for k,v in fields.items()})
