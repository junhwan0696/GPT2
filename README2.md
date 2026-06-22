# TinyGPT — Character-level Language Model

## 코드 진행 설명
1. 데이터 준비
   텍스트 파일을 읽어온 후 string을 integer로 변환해서 저장한다. 이를 DataLoader를 통해 batch_size씩 만큼 나눠 담는다.
   
2. multi-head attention
   Head 클래스는 받아온 데이터에서 attention 계산을 위해 필요한 key, query, value 값을 뽑아낸다. 이후 논문에서 주어진 attention 값을
   게산한다.
   $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$  
   MultiHeadAttention 클래스에서는 Head 클래스에서 만든 요소들을 여러개 엮어서 multi-head attention을 만든다.  
   이를 통해 단어들 사이의 문맥(관련성)을 파악한다.
   
3. Feedforward와 Block
   FeedForward 클래스를 통해 입력 차원을 4배로 늘린 후 nn.ReLU()(<- 음수는 0으로, 양수는 그대로)를 통해 비선형적인  
   패턴도 학습하게 하고 다시 1/4로 줄인다. 이 과정을 통해 성능이 향상된다.  
   Block은 MultiHeadAttention과 FeedForward를 합친다.
   
4. TinyGPT
   TinyGPT는 앞에서 만들어둔 것을 합친 최종적인 언어모델이다. 이 모델에서 일어나는 일을 정리하면  
   (1) 입력된 테이터 토큰을 embedding 한다.(문자를 숫자로 변환)  
   (2) 문장 내에서의 위치정보도 embedding 한다.  
   (3) block(MultiHeadAttention + FeedForward)을 여러 개로 쌓는다.  
   (4) 의미와 문장 내에서의 순서 백터가 합쳐져서 입력 벡터로 쓰이게 된다.  
   (5) 최종적으로 계산 후 다음에 어떤 글자가 나올지 보는 점수 행렬(logit)을 return한다.
   
5. 학습
   F.cross_entropy가 예측과 정답을 비교해 loss 함수를 계산한다.  
   train_one_epoch 함수를 통해 1 epoch를 학습시킨다. 이때, 1 batch를 학습할 때마다 gradient를 초기화한다.  
   Backpropagation을 통해 역추적하면서 gradient를 계산한다.  
   이후 optimizer를 통해 계산된 gradient 방향으로 가중치를 조금씩 변경한다.  
   loss 값을 확인하기 위해 기록해둔다.
   
6. sampling
   @torch.no_grad()를 통해 gradient를 계산하지 않도록 한다.(학습 과정이 아니기에 필요 없음)  
   start_text를 주면 이것을 맨 오른쪽에 두고 나머지는 0으로 채운 context를 만들고 model에 넣어서 다음 단어에  
   대한 logit을 구하게 한다. 그리고 맨 마지막 단어에 대한 logit만 추출해서 torch.multinomial을 활용해 구한  
   확률분포에 기반해 다음 글자를 구하고 이를 결과 list와 context에 업데이트 한다.  
   
---

## 모델 아키텍처

```
Input tokens
    └─ Token Embedding + Position Embedding
           └─ Block × num_layers
                  ├─ LayerNorm → MultiHeadAttention → Residual
                  └─ LayerNorm → FeedForward       → Residual
           └─ LayerNorm (final)
           └─ LM Head (Linear → vocab_size)
Output logits


```

## 주요 클래스

### `NextTokenDataset`
PyTorch `Dataset`을 상속한 데이터셋 클래스.

| 인자 | 설명 |
|---|---|
| `data` | 토큰 인덱스 텐서 (전체 텍스트를 정수로 인코딩) |
| `block_size` | 입력 시퀀스 길이 (컨텍스트 윈도우 크기) |

- `__getitem__`: 인덱스 `idx`에서 길이 `block_size`의 입력 `x`와, 1칸 이동한 타깃 `y`를 반환

---

### `Head`
단일 어텐션 헤드. Scaled Dot-Product Attention을 구현합니다.

| 멤버 | 설명 |
|---|---|
| `self.key` | 입력을 key 공간으로 투영 (`emb_dim → head_size`, bias 없음) |
| `self.query` | 입력을 query 공간으로 투영 (`emb_dim → head_size`, bias 없음) |
| `self.value` | 입력을 value 공간으로 투영 (`emb_dim → head_size`, bias 없음) |
| `self.tril` | 미래 토큰을 가리는 causal mask (하삼각 행렬) |
| `self.dropout` | 어텐션 가중치에 적용하는 드롭아웃 |

**forward 흐름:**
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

### `MultiHeadAttention`
`Head`를 `num_heads`개 병렬로 실행 후 결과를 concat하여 출력 투영합니다.

| 멤버 | 설명 |
|---|---|
| `self.heads` | 독립적인 `Head` 인스턴스 리스트 |
| `self.proj` | concat 결과를 원래 `emb_dim`으로 투영하는 선형 레이어 |
| `self.dropout` | 출력 드롭아웃 |

---

### `FeedForward`
각 위치(position)에 독립적으로 적용되는 2층 MLP.

- 구조: `emb_dim → 4×emb_dim (ReLU) → emb_dim (Dropout)`
- Transformer의 "position-wise FFN"에 해당

---

### `Block`
Transformer 디코더 블록. Pre-LayerNorm 방식으로 안정적인 학습을 지원합니다.

| 멤버 | 설명 |
|---|---|
| `self.ln1` | MultiHeadAttention 전 LayerNorm |
| `self.sa` | MultiHeadAttention |
| `self.ln2` | FeedForward 전 LayerNorm |
| `self.ffwd` | FeedForward |

**Residual connection:**
```
x = x + sa(ln1(x))
x = x + ffwd(ln2(x))
```

---

### `TinyGPT`
전체 GPT 모델 클래스.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `vocab_size` | — | 문자 어휘 크기 |
| `block_size` | — | 최대 시퀀스 길이 |
| `emb_dim` | 128 | 임베딩 차원 수 |
| `num_heads` | 4 | 어텐션 헤드 수 |
| `num_layers` | 4 | Transformer 블록 수 |
| `dropout` | 0.1 | 드롭아웃 비율 |

| 멤버 | 설명 |
|---|---|
| `self.token_embedding` | 토큰 인덱스를 `emb_dim` 벡터로 변환 |
| `self.position_embedding` | 위치 인덱스를 `emb_dim` 벡터로 변환 |
| `self.blocks` | `num_layers`개의 `Block`을 순서대로 실행 |
| `self.ln_f` | 최종 출력 전 LayerNorm |
| `self.lm_head` | `emb_dim → vocab_size` 선형 투영 (logits 출력) |

---

## 주요 함수

### `sequence_cross_entropy(logits, targets)`
시퀀스 단위 Cross-Entropy 손실 계산.

- `logits`: shape `(B, T, vocab_size)`
- `targets`: shape `(B, T)`
- `logits.transpose(1, 2)`로 shape을 `(B, vocab_size, T)`로 변환하여 `F.cross_entropy`에 전달

---

### `train_one_epoch(model, loader, optimizer, device, max_steps=None)`
1 epoch 학습을 수행하고 평균 손실을 반환합니다.

| 인자 | 설명 |
|---|---|
| `model` | 학습할 TinyGPT 모델 |
| `loader` | DataLoader |
| `optimizer` | AdamW optimizer |
| `device` | `"cuda"` 또는 `"cpu"` |
| `max_steps` | 한 epoch 내 최대 배치 수 (None이면 전체 사용) |

---

### `sample_gpt(model, block_size, stoi, itos, device, start_text, max_new_tokens)`
학습된 모델로 텍스트를 생성합니다.

| 인자 | 설명 |
|---|---|
| `start_text` | 생성 시작 텍스트 (기본값: `"보고싶어"`) |
| `max_new_tokens` | 생성할 최대 문자 수 |

- `start_text`의 각 문자를 컨텍스트에 순서대로 삽입
- 매 스텝마다 마지막 위치의 logits에 softmax를 적용해 다음 문자를 샘플링
- 슬라이딩 윈도우 방식으로 컨텍스트를 유지 (`block_size` 고정)

---

## 주요 변수

| 변수 | 설명 |
|---|---|
| `stoi` | 문자 → 인덱스 딕셔너리 |
| `itos` | 인덱스 → 문자 딕셔너리 |
| `vocab_size` | 고유 문자 수 |
| `block_size` | 컨텍스트 윈도우 크기 (기본값: 64) |
| `data` | 전체 텍스트를 인덱스 텐서로 변환한 것 |
| `device` | 학습 장치 (`cuda` 우선, 없으면 `cpu`) |

---

## 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (lr=3e-4) |
| Epochs | 200 |
| Max steps/epoch | 300 |
| Batch size | 64 |

---

## 실제 트레이닝 결과

친구와의 카카오톡 대화 내역을 텍스트 파일로 추출하여 트레이닝해보았습니다. start_text를 "점심 뭐 먹을래"로 하고 sampling한 결과입니다.
<img width="1630" height="557" alt="image" src="https://github.com/user-attachments/assets/e8b9e7ee-f91e-4162-b1cc-f84292f4f335" />

```
