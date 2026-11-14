"""모델 호출 담당. 결과를 표 형태로 바꾸며 test를 입력하지 않습니다."""
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

def detect_images(weights,folder,output,conf=.25):
    model=YOLO(str(weights));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];counts=[]

    print("모델 클래스:", model.names)
    # TODO C07: folder를 stream=True로 추론합니다. 결과 객체를 전부 메모리에 모으지 않습니다.
    # model.predict(): 이미지·영상에서 물체를 탐지하는 메서드. 정답 라벨 없이 예측하므로 TP·FP·mAP는 계산하지 않음
    # stream: 예측 결과를 어떤 방식으로 받을지 정하는 옵션
    # stream=True: 전체 결과를 리스트로 모아 두지 않고, 필요한 결과를 하나씩 생성해서 전달. 이전 결과를 계속 보관하지 않아 메모리를 절약할 수 있음. 현재 처리 중인 프레임과 결과에는 메모리를 사용
    '''stream=True / False 동작 차이
    
    results = model.predict(source="demo.avi", stream=False)
    
    # 여기서 전체 영상 추론 → 결과를 리스트에 저장 → 반환

    for result in results:
    print(result.boxes)  # 이미 계산된 결과를 출력

    전체 프레임 추론 완료 → 리스트 반환 → 결과 출력
    ----------------------------------------------

    results = model.predict(source="demo.avi", stream=True)
    
    # generator 반환. 아직 추론은 시작하지 않음

    for result in results:
        # 반복하면서 추론이 진행되고 결과를 받음
        print(result.boxes)
    
    프레임 추론 → 결과 출력 → 다음 프레임 추론 → 결과 출력 → …

    stream=True에서는 for문으로 순회해야 추론이 진행

    '''    
    results = model.predict(
        source=str(folder),
        conf=conf,
        imgsz=160,
        device="cpu",
        stream=True,
        project=out,
        name="predict",
        verbose=False
    )

    for r in results:
        image=Path(r.path).name;counts.append(dict(image=image,detections=len(r.boxes)))
        cv2.imwrite(str(out/image),r.plot())
        for box,cls,score in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist()):
            rows.append([image,int(cls),score,*box])
    return pd.DataFrame(rows,columns=["image","class_id","confidence","x1","y1","x2","y2"]),pd.DataFrame(counts)

def track_video(weights,video,output,tracker_name='bytetrack.yaml'): 
    model=YOLO(str(weights));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    # cv2.CAP_PROP_FPS: 초당 프레임 수
    # cv2.CAP_PROP_FRAME_WIDTH: 프레임 가로 크기(px)
    # cv2.CAP_PROP_FRAME_HEIGHT: 프레임 세로 크기(px)
    fps=cap.get(cv2.CAP_PROP_FPS);size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)));cap.release()
    
    # fourcc: 프레임을 인코딩하는 방식
    writer=cv2.VideoWriter(
        filename=str(out/'tracked.avi'),
        fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
        fps=fps,
        frameSize=size
    )
    rows=[];frames=[]
    # TODO C08: 새 영상용 모델에 tracker_name을 명시해 추적합니다. persist=False로 시작합니다.
    '''
        영상에서도 목적에 따라 predict()와 track()을 선택

        # 각 프레임에서 무엇이 어디 있는지 탐지
        results = model.predict(source="demo.avi", stream=True)

        # 같은 물체가 어디로 움직이는지 추적
        results = model.track(source="demo.avi", stream=True)

        예를 들어 사람이 두 명 있을 때,
        predict()는 매 프레임 “사람 2명”을 탐지하고,
        track()은 “ID 1번 사람, ID 2번 사람”을 구분하며 따라감

        즉, 영상이라서 반드시 track()을 쓰는 것은 아니고, 물체의 ID를 이어서 관리해야 할 때 사용

        ------------------------

        -stream 여부와 관계없이 track은 프레임 사이의 물체를 연결해 추적

        -persist: 호출 사이에 추적 상태를 유지할지
                  새 모델로 영상 하나를 한 번만 처리한다면 persist 차이는 사실상 없음
                  같은 모델로 이전에 추적한 이력이 있다면, 상태 재사용 여부에 영향을 줄 수 있음

                  model.track(frame1, persist=True)
                  model.track(frame2, persist=True)
                  이 경우 persist=True를 사용해야 이전 프레임의 추적 상태를 이어갈 수 있음
        
        -tracker: 프레임 사이에서 같은 물체를 연결할 추적 알고리즘과 설정을 선택하는 역할
    '''

    # 호출횟수 뿐 아니라 입력 파일 경로가 바뀌는지에 따라서도 persist=True 필요 달라짐
    # 영상 하나를 한 번 호출로 처리     persist=False로 가능
    # JPG 폴더를 한 번 호출로 처리      persist=True <-- 입력파일 경로가 바뀌므로.
    # 프레임마다 track()을 반복 호출	persist=True
    results=model.track(
        source=str(video),
        stream=True,
        persist=False,
        tracker=tracker_name,
        conf=.25,
        imgsz=160,
        device="cpu",
        verbose=False
    )

    # r에는 박스 좌표, 클래스 번호, 탐지 신뢰도, 추적ID 등이 담겨있음
    # 한 프레임에 물체가 여러 개 있으면 각 속성에도 물체별 값이 들어 있음
    # 추적 ID가 부여되지 않은 경우 r.boxes.id는 None일 수 있음 - 탐지가 없거나 추적이 아직 확정되지 않은 경우
    # r.plot()은 박스·클래스·추적 ID 등을 그린 이미지 배열을 반환
    # [Q] 추적 ID는 언제 부여?
    # [A] 추적기가 탐지된 물체를 새로운 추적 대상으로 등록할 때 ID를 부여해요.
    #     흐름은 다음과 같아요.
    #       1. YOLO가 현재 프레임에서 물체를 탐지해요.
    #       2. 추적기가 기존에 추적하던 물체와 비교해요.
    #       3. 기존 물체와 연결되면 기존 ID를 유지해요.
    #       4. 새 물체로 인정되면 새 ID를 부여해요.
    #     단, 모든 탐지에 즉시 ID가 표시되는 것은 아니에요. 신뢰도 기준과 추적기의 확인 조건을 충족해야 결과에 ID가 나올 수 있어요.
    #     물체를 놓쳤다가 다시 발견했을 때도, 기존 물체로 연결되면 같은 ID를 유지하고 연결에 실패하면 새 ID가 생길 수 있어요.
    for frame,r in enumerate(results):
        writer.write(r.plot());frames.append([frame,len(r.boxes)])
        if r.boxes.id is not None:
            for box,cls,tid in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.id.cpu().tolist()):
                rows.append([frame,int(tid),int(cls),*box])
    writer.release()
    return pd.DataFrame(rows,columns=['frame','track_id','class_id','x1','y1','x2','y2']),pd.DataFrame(frames,columns=['frame','detections'])
