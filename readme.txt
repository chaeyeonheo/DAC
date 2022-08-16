With CNU Energy+AI

[주제] 딥러닝 기반 불법촬영목적 드론 실시간 감지 시스템

[파일 및 폴더 설명]
0. runs : 실행결과를 저장할 폴더
1. DaC_realtime_video.py : 실시간 감지 시스템이 가능하게끔 구현된 Yolov5+VGG Code
2. bast.pt : 11,000장의 데이터를 학습시킨 도중에 가장 학습이 잘된 model
3. vgg11.pth : vgg11을 실행시키기 위한 model(parameter 포함됨)

[실행방법]
0. Webcam 준비
1. Pytorch 기반의 Yolov5 환경 구축
	link => https://github.com/ultralytics/yolov5
	Quick Start Examples 따라하기
	*** pytorch 기반 가상환경을 만드는 것을 추천합니다. 환경이 망가질 수도 있습니다. ㅠ_ㅠ
2. 터미널 열어서 다음과 같은 명령어 입력
	cd ./DaC_CloseCV/
	conda activate pytorch
	python DaC_realtime_video.py --source 0 --sace-crop --save-txt --weight best.pt --conf 0.5
3. realtime으로 실행시키기
4. 종료는 q 혹은 ctrl+s
5. runs folder에 저장된 결화 확인해보기


VGG11 model 만들기나, 이미지 합성관련 코드를 다운받고 싶으시다면 GitHub에 놀러오세요!
	link => 
