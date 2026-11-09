"""실습 02: Tesseract를 실제 호출합니다. 엔진과 eng 언어팩이 필요합니다."""
from pathlib import Path
import os
import pandas as pd
import pytesseract
import cv2

''' tesseract 엔진 설정

# PATH에 C:\Program Files\Tesseract-OCR를 추가했다면 아래 코드는 생략할 수 있습니다.
# PATH에 설치여부 확인:
# 1) tesseract --version
# 2) tesseract --list-langs

# Windows에서 PATH 설정이 어려우면 환경변수 TESSERACT_CMD에 exe 경로를 지정합니다.

# 실제 설치 위치에 맞게 수정
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
혹은
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')

print(pytesseract.get_tesseract_version())
'''

#from PIL import Image
#text = pytesseract.image_to_string(Image.open("image.png"), lang="kor+eng")
#print(text)

base = Path(__file__).resolve().parent

paths = [str(base / f'doc{i:02d}.png') for i in range(1, 4)]
input_list = base / 'p02_inputs.txt'
input_list.write_text('\n'.join(paths), encoding='utf-8')

results = []
# TODO B02:  여러 페이지를 목록으로 전달해 엔진을 한 번 호출합니다.
words = pd.DataFrame(columns=['page_num', 'text', 'conf'])
for path in paths:
    '''
    단순히 이미지 하나의 글자를 읽어 text 로 반환: image_to_string

    img = cv2.imread(path)

    text = pytesseract.image_to_string(img, lang="eng")
    words.loc[len(words)] = []
    results.append({
        "path": path,
        "text": text.strip()
    })

# 결과 출력
for result in results:
    print(f"\n파일: {result['path']}")
    print(result["text"])
'''
    pytesseract.image_to_data(
        str(input_list),
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )


words = words.dropna(subset=['text'])
words = words[words['conf'] >= 0]
words.to_csv(base / 'p02_words.csv', index=False, encoding='utf-8-sig')
for page, rows in words.groupby('page_num', sort=True):
    text = ' '.join(rows['text'].astype(str))
    print('page:', page, 'words:', len(rows), 'mean confidence:', round(rows['conf'].mean(), 2))
    print(text)
print('backend: live Tesseract; confidence는 정답 확률이 아닙니다.')
