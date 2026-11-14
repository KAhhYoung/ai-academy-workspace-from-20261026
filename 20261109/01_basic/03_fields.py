"""실습 03: 실제 OCR 기록에서 필요한 3개 필드를 추출합니다. 정상 자료만 사용합니다."""
from pathlib import Path
import json
import re

base = Path(__file__).resolve().parent
docs = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))['documents']
records = []
for name in ['doc01', 'doc02', 'doc03']:
    text = docs[name]['ocr']['text']
    # TODO B03: DATE 줄 뒤의 날짜 부분을 추출합니다. OCR 원문도 함께 보존합니다.
    date_pat = r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})"
    date_match = re.search(date_pat, text)
    issued_at = date_match.group(0) if date_match else None

    supplier = re.search(r'^SUPPLIER: (.+)$', text, re.MULTILINE).group(1)
# TODO B04
    match = re.search(r'^TOTAL: (.+)$', text, re.MULTILINE)
    total_amount = match.group(1).strip() if match else None

    records.append({'document_id': name, 'issued_at': issued_at, 'supplier': supplier, 'total_amount': total_amount, 'currency': 'KRW', 'ocr_source': 'recorded Tesseract', 'raw_text': text})
    print(name, issued_at, supplier, total_amount)
(base / 'p03_fields.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
