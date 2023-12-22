# KWU-Analysis-Model
Deep learning model for physiological time-series analysis

## Content
### Analysis-Model (Discriminative Graph Transformer Model)
#### Produced by NeuroAI Lab. @Kwangwoon Univ.

![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/baff03da-0fca-4b2e-a8ea-ab6382a8ff6d){: .align-center}
*Discriminative Graph Transformer Model (DTGM)*

*	EEG 감정인식 모델은 Graph Encoder Module (GEM), 그리고 Graph Transformer Module (GTM)으로 구성된다.
*	GEM은 두개의 GCN으로 구성되며, GCN은 각 노드의 특징을 직접 연결된 주변 노드들로부터 엣지 가중치 (유사성 점수)와 곱해진 특징을 전파 받아 고수준의 특징으로 합성한다. 유사한 노드끼리 특징 합성이 이루어지므로 클래스 별 특징 군집을 뚜렷하게 구성할 수 있다는 장점이 있지만, 직접 연결된 노드의 특징만 고려하므로, 국소적인 이웃 관계에 고착된다.
* 이 문제를 보완하고자, 그래프 트랜스포머를 도입하였으며, 기존 어텐션 스코어 (cosine 유사성) 계산 시 엣지 정보 (유클리드 거리 기반 유사성)를 추가로 활용함으로써 그래프 전체적 이웃 관계를 고려한다.
*	Classifier로는 하나의 Fully-connected (FC) layer를 활용한다.

![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/fddc4631-3b5f-4ffa-bb65-9630a1a1c787){: .align-center}
*Graph Convolution*
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/e72caeb8-e972-41ee-81f8-94153514f012){: .align-center}
*Graph Multi-head Attention*


## Used signals

### Electroencephalogram (EEG)
![image](https://github.com/ClustProject/KWUAnalysisModels/assets/74770095/2a3ac3d9-cee3-426d-9af3-1492e9b0ea59){: .align-center}

*	뇌전도를 기록하기 위해서 전극을 두피에 부착하며, 전극의 수에 따라 뇌전도 신호의 채널 수가 정해진다.
*	뇌전도 각 채널로부터 특정 뇌 지역의 활동을 알 수 있으며, 해당 뇌 지역이 활성화되는 경우 스파이크와 함께 복잡한 파형이 기록된다.
*	심전도, 심탄도 등의 심장 신호와 달리 반복되는 파형이 기록되지 않고 다른 생체 신호보다 복잡한 파형을 가지기 때문에 정밀하고 다양한 분석 기법이 필요하다.
*	뇌전도는 총 5개의 주파수 대역 (δ (1-3 Hz) wave, θ (4-7 Hz) wave, α (8-13 Hz) wave, β (14-30 Hz) wave, γ (31-50 Hz) wave)으로 나누어 질 수 있다.




