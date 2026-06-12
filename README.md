# GPT2
GPT2를 구현한 코드입니다.


## 📌 프로젝트 소개

이 프로젝트는 PyTorch를 활용하여 작은 규모의 GPT(Generative Pre-trained Transformer) 모델을 바닥부터(from scratch) 구현한 것입니다. Multi-head attention, Feedforward network, Residual connection, Layer normalization 등의 핵심 트랜스포머(Transformer) 구조를 단계별로 조립하여 완성했습니다.  특정 텍스트 데이터셋을 학습하여 문맥을 파악하고 새로운 문장을 생성할 수 있습니다.

---

## 핵심 클래스 및 함수 역할

### 1. 데이터 파이프라인 (Data Processing)

* 
**`NextTokenDataset (Dataset)`**: 모델이 학습할 수 있는 형태로 텍스트 데이터를 변환하는 커스텀 PyTorch Dataset 클래스입니다. 


* 
**`__getitem__`**: 지정된 `block_size`만큼의 길이로 입력 시퀀스(x)와 한 칸씩 밀린 정답 시퀀스(y, 예측해야 할 다음 토큰)를 잘라내어 반환합니다. 





### 2. 어텐션 메커니즘 (Attention Mechanism)

* 
**`Head (nn.Module)`**: 모델이 문맥을 파악하는 핵심인 Masked Self-Attention을 수행하는 단일 헤드입니다. 


* 
**`forward`**: 입력된 텐서로부터 Key, Query, Value 행렬을 계산합니다. 이후 하삼각행렬(`tril`)을 마스크로 씌워 모델이 미래의 토큰을 부정행위처럼 미리 보지 못하게 차단한 뒤, 소프트맥스(Softmax)를 거쳐 어텐션 스코어를 산출합니다. 




* 
**`MultiHeadAttention (nn.Module)`**: 여러 개의 `Head`를 병렬로 연결하여, 모델이 텍스트의 다양한 특징을 동시에 학습할 수 있도록 돕는 클래스입니다. 


* 
**`forward`**: 개별 헤드들이 계산한 결괏값들을 하나로 합친(concatenate) 후, 선형 변환(Projection)과 Dropout을 적용하여 최종 출력합니다. 





### 3. 트랜스포머 블록 (Transformer Block)

* 
**`FeedForward (nn.Module)`**: 어텐션 연산 뒤에 위치하여 데이터를 비선형적으로 처리하는 단순한 형태의 다층 퍼셉트론(MLP)입니다. 


* 
**`forward`**: 선형 계층(Linear Layer)과 ReLU 활성화 함수를 통과시키며 모델의 표현력을 높여줍니다. 




* 
**`Block (nn.Module)`**: 앞서 만든 Multi-Head Attention과 FeedForward를 하나로 묶은 트랜스포머의 기본 빌딩 블록입니다. 


* 
**`forward`**: 입력값이 Layer Normalization을 거쳐 어텐션과 FeedForward를 통과할 때마다 잔차 연결(Residual Connection, `x = x + ...`)을 더해주어, 깊은 신경망에서도 기울기 소실 문제 없이 안정적으로 학습되도록 구성되었습니다. 





### 4. 메인 모델 (Main Language Model)

* 
**`TinyGPT (nn.Module)`**: 여러 개의 `Block`을 층층이 쌓아 올려 완성된 최종 GPT 모델입니다. 


* 
**`forward`**: 입력된 단어 인덱스를 Token Embedding과 Position Embedding으로 변환해 더해줍니다. 이후 여러 겹의 트랜스포머 블록과 마지막 Layer Norm, Linear 계층을 차례로 통과시켜, 단어 사전(`vocab_size`) 크기만큼의 예측값(logits)을 반환합니다. 





### 5. 학습 및 텍스트 생성 (Training & Sampling)

* 
**`sequence_cross_entropy`**: 모델의 예측값(logits)과 실제 정답(targets)의 형태를 맞추어 Cross Entropy 손실(Loss)을 계산하는 함수입니다. 


* 
**`train_one_epoch`**: 모델을 1 에포크 동안 실제로 학습시키는 훈련 루프입니다. 


* 데이터 로더로부터 배치를 받아 손실을 계산하고, 역전파(Backpropagation, `loss.backward()`)와 옵티마이저(`optimizer.step()`)를 통해 모델의 가중치를 업데이트합니다. 




* 
**`sample_gpt`**: 학습이 끝난 모델을 이용해 새로운 문장을 만들어내는(추론) 함수입니다. 


* 초기 텍스트(`start_text`)를 입력받은 뒤, 모델이 출력하는 확률 분포에 기반하여 `torch.multinomial`로 다음 글자를 샘플링하고, 이를 반복하여 문장을 길게 이어나갑니다. (가중치 업데이트를 방지하기 위해 `@torch.no_grad()[cite_start]`가 적용되었습니다.) 
