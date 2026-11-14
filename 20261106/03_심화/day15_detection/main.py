"""실습 07. 데이터 감사와 실제 YOLO 추론·추적을 연결합니다."""
from pathlib import Path
import json,hashlib,shutil
import torch
from vision import detect_images,track_video
BASE=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(1)
    out=BASE/'outputs';out.mkdir(exist_ok=True)
    #weight=BASE/'data/demo.pt'
    weight=BASE/'data/best_smoke_13.pt'
    predictions,counts=detect_images(weights=weight,folder=BASE/'data/images/val',output=out/'images')
    tracks,frames=track_video(weights=weight,video=BASE/'data/demo.avi',output=out)
    predictions.to_csv(out/'predictions.csv',index=False);counts.to_csv(out/'image_counts.csv',index=False)
    tracks.to_csv(out/'tracks.csv',index=False);frames.to_csv(out/'frames.csv',index=False)
    # TODO C09: 실제 처리한 이미지 수·프레임 수·모델 해시를 기록하세요.
    summary=dict(images=len(counts),frames=len(frames),detections=len(predictions),track_rows=len(tracks),model_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),split='val',test_used=False)


    shutil.copyfile(src=weight,dst=out/'model.pt')  # 신뢰하는 자체 준비 모델을 함께 보관합니다.

    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

# 사전학습된 demo.pt 를 추가학습 없이 사용하여서는 탐지,추적 없음
# 새로운 데이터로 학습하여 얻은 best.pt(epoch=250) 로 predict 및 track 수행
