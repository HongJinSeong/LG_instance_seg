# LG_instance_seg
LG instance segmentation


mmsegmentation을 통한 2번째 대회 

instance segmentation 이여서 좌표값을 내야했음

다양한 augmentation , model 사용하는 것으로 진행함 

최종적으로는 SCNet에 augmenatation (CopyPaste / RandomFlip / CutOut / RandomShift / RandomCrop) 적용하여 진행 

최종 8등/62

n-fold ensemble을 진행해서 좀더 general 한 결과를 내보도록 해야 했는데 data의 balance 부분은 전혀 생각하지 않고 overfitting이 되어서 높은 성능을 거두지 못한 것으로 보임

적정한 data split으로 의미 있는 validation 기준을 잡았어야 하는데 한번 데이터 나눈기준을 계속 가져감으로써 좋은 결과 못얻었음

validation과 test의  결과가 크게 차이 날 때는 현재의 기준의 적정한가 다시 한번 데이터 기준으로 살펴볼 필요가 있다고 생각되었음
