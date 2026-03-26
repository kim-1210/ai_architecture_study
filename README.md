# ai_architecture_study

AI 모델 구조를 공부하면서 직접 구현하고, 가능한 경우 학습 결과까지 같이 정리해둔 저장소입니다. 현재는 `transformer`, `ddpm`, `flow_matching`, `diffusion_policy` 네 파트로 구성되어 있습니다.

## 개요

- `pyproject.toml`
  - 프로젝트 의존성을 관리합니다.
  - 주요 패키지는 `torch`, `torchvision`, `numpy`, `tqdm`, `gymnasium[mujoco]` 입니다.

- `uv.lock`
  - 의존성 버전 고정 파일입니다.

- `main.py`
  - 현재는 간단한 실행 확인용 파일입니다.
  - 실제 실험 코드는 각 하위 디렉터리에 정리되어 있습니다.

## 1. transformer

Transformer 구조를 직접 구현해본 파트입니다.

- `transformer/model.py`
  - embedding, positional encoding, multi-head attention, encoder/decoder block 등 기본 구성 요소를 직접 구현했습니다.

- `transformer/train.ipynb`
  - Transformer 학습용 노트북입니다.
  - 데이터 로딩, 설정값 정의, 학습 루프 구성이 들어 있습니다.

- `transformer/tokenizer_english.json`
  - 영어 토크나이저입니다.
  - vocab 크기는 `30000`입니다.

- `transformer/tokenizer_korean.json`
  - 한국어 토크나이저입니다.
  - vocab 크기는 `30000`입니다.

### 상태

Transformer는 구현 위주로 정리되어 있습니다. 모델 코드와 토크나이저는 포함되어 있지만, 학습 결과 이미지나 체크포인트는 현재 저장소에 정리되어 있지 않습니다.

## 2. ddpm

TinyHero 데이터셋으로 기본 DDPM을 학습해본 파트입니다.

- `ddpm/ddpm_model.py`
  - DDPM용 U-Net 계열 모델 정의 파일입니다.
  - residual block, down/up block, time embedding 등이 포함되어 있습니다.

- `ddpm/ddpm_train.ipynb`
  - 기본 DDPM 학습 노트북입니다.
  - `64x64` 이미지 기준으로 학습하고, 매 epoch마다 샘플 이미지를 저장하도록 구성되어 있습니다.

- `ddpm/tinyhero.zip`
  - TinyHero 데이터 압축 파일입니다.

- `ddpm/datas/tinyhero/`
  - 실제 학습 이미지가 들어 있는 디렉터리입니다.
  - PNG 이미지 `3648`장을 사용합니다.

- `ddpm/ddpm_img/`
  - 학습 중 저장한 샘플 이미지입니다.

- `ddpm/weight/ddpm.pth`
  - 최종 DDPM weight입니다.

### 결과

- 노트북 로그 기준으로 `300` epoch 학습을 수행했습니다.
- 첫 epoch 로그에 `Loss=0.119`가 기록되어 있습니다.
- `ddpm/ddpm_img`에는 `ddpm_epoch_001.png`부터 `ddpm_epoch_299.png`까지 샘플 이미지가 저장되어 있습니다.
- 최종 weight는 `ddpm/weight/ddpm.pth`에 저장되어 있습니다.

## 3. latent diffusion + vae

같은 `ddpm` 디렉터리 안에서 VAE를 먼저 학습한 뒤, latent space 상에서 diffusion을 학습한 실험입니다.

- `ddpm/ldpm_model.py`
  - VAE 모델 정의 파일입니다.
  - encoder, decoder, reparameterization, KL loss 계산이 포함되어 있습니다.

- `ddpm/ldpm_train.ipynb`
  - VAE 학습과 latent diffusion 학습을 함께 수행하는 노트북입니다.
  - 설정값 기준 `vae_n_epoch=2000`, `n_epoch=7000`, `latent_dim=128`로 구성되어 있습니다.

- `ddpm/vae_img/`
  - VAE reconstruction 결과 이미지입니다.

- `ddpm/ldpm_img/`
  - latent diffusion 샘플 이미지입니다.

- `ddpm/ldpm_weight/`
  - VAE 및 latent diffusion weight가 저장된 디렉터리입니다.

### 결과

- VAE 첫 epoch 로그에는 `loss=0.2356`, `recon=0.2354`, `kl=0.1839`, `beta=0.001`가 기록되어 있습니다.
- `ddpm/vae_img`에는 `vae_epoch_060.png`부터 `vae_epoch_1980.png`까지 결과 이미지가 저장되어 있습니다.
- latent diffusion 첫 epoch 로그에는 `Loss=1.02`가 기록되어 있습니다.
- `ddpm/ldpm_img`에는 `ldm_epoch_100.png`부터 `ldm_epoch_1300.png`까지 샘플 이미지가 저장되어 있습니다.
- `ddpm/ldpm_weight`에는 `vae.pth`, 중간 VAE 체크포인트, `ldm_100epoch.pth`부터 `ldm_1300epoch.pth`까지의 diffusion 체크포인트가 함께 저장되어 있습니다.

## 4. flow matching

TinyHero 이미지 데이터를 대상으로 flow matching 기반 생성 모델을 학습한 파트입니다.

- `flow_matching/flow_matching_unet_model.py`
  - flow matching에서 사용하는 U-Net 계열 모델 정의 파일입니다.
  - `ResidualConvBlock`, `UnetDown`, `UnetUP`, `EmbedFC`, `ContextUnet`으로 구성되어 있습니다.
  - 시간 정보와 context 입력을 각각 임베딩해서 업샘플링 경로에 주입하는 구조입니다.

- `flow_matching/flow_matching_train.ipynb`
  - flow matching 학습 노트북입니다.
  - TinyHero 이미지를 불러와 `64x64`로 resize 후 `[-1, 1]` 범위로 정규화해 학습합니다.
  - 학습 중 `x_t = t * x + (1 - t) * noise` 형태로 interpolation을 만들고, 모델은 `x - noise`에 해당하는 vector field를 예측하도록 구성되어 있습니다.

- `flow_matching/flow_matching_img/`
  - 매 epoch 저장된 샘플 이미지입니다.

- `flow_matching/flow_matching_weight/`
  - 최종 및 중간 체크포인트가 저장된 디렉터리입니다.

### 학습 설정

노트북 설정 기준 주요 하이퍼파라미터는 다음과 같습니다.

- 데이터 경로: `../ddpm/datas/tinyhero`
- 데이터셋 크기: `3648`
- 이미지 크기: `64x64`
- 채널 수: `3`
- 배치 크기: `100`
- epoch 수: `300`
- learning rate: `1e-3`
- ODE integration step 수: `500`
- feature width: `128`
- context feature 차원: `5`

### 결과

- 노트북 출력 기준 데이터로더 길이는 `37`, 데이터셋 크기는 `3648`입니다.
- 첫 epoch 로그에는 `Loss=0.0889`가 기록되어 있습니다.
- 학습은 `300` epoch까지 수행되었습니다.
- `flow_matching/flow_matching_img`에는 `flow_matching_epoch_001.png`부터 `flow_matching_epoch_300.png`까지 총 `300`장의 샘플 이미지가 저장되어 있습니다.
- `flow_matching/flow_matching_weight`에는 최종 weight `flow_matching.pth`와 함께 `30` epoch 단위 중간 체크포인트가 저장되어 있습니다.
  - `flow_matching_30epoch.pth`
  - `flow_matching_60epoch.pth`
  - `flow_matching_90epoch.pth`
  - `flow_matching_120epoch.pth`
  - `flow_matching_150epoch.pth`
  - `flow_matching_180epoch.pth`
  - `flow_matching_210epoch.pth`
  - `flow_matching_240epoch.pth`
  - `flow_matching_270epoch.pth`
  - `flow_matching_300epoch.pth`

정리하면 이 파트는 단순 구현에 그치지 않고, 모델 정의, 학습 노트북, 중간 샘플 이미지, 체크포인트까지 모두 포함된 상태입니다.

## 5. diffusion_policy

이미지 생성이 아니라 observation으로부터 미래 action chunk를 예측하는 diffusion policy 실험입니다.

- `diffusion_policy/based_cnn_model.py`
  - 1D convolution 기반 diffusion policy 모델 정의 파일입니다.
  - timestep embedding과 observation conditioning을 함께 사용합니다.

- `diffusion_policy/process_data.py`
  - 원본 rollout 데이터를 학습용 action chunk 형태로 변환하는 전처리 코드입니다.
  - 각 시점 observation에 대해 앞으로 `horizon`만큼의 action을 잘라 데이터셋을 구성합니다.

- `diffusion_policy/base_cnn_train.ipynb`
  - 전처리된 데이터를 이용해 실제 학습과 샘플링을 수행하는 노트북입니다.

- `diffusion_policy/datas/reach_bc.npz`
  - 원본 rollout 데이터입니다.

- `diffusion_policy/datas/reach_bc_imitation_h15.npz`
  - 전처리 후 생성된 데이터입니다.
  - 파일명 그대로 `horizon=15` 기준 데이터입니다.

### 결과

- 전처리 후 데이터 shape은 다음과 같습니다.
  - `observations`: `(1000000, 32)`
  - `action_chunks`: `(1000000, 15, 7)`
  - `action_chunks_t`: `(1000000, 7, 15)`

- 학습 노트북 출력 기준:
  - 데이터셋 크기: `1,000,000`
  - 배치 shape: `torch.Size([64, 32])`, `torch.Size([64, 7, 15])`
  - 모델 파라미터 수: `207367`
  - 첫 epoch 로그: `epoch=001 mean_loss=1.302935`

- 노트북 마지막에는 `ground truth chunk[0]`와 `sampled chunk[0]`를 함께 출력해 샘플링 결과를 비교할 수 있게 해두었습니다.

## 정리

이 저장소는 단순히 예제 코드를 모아둔 형태보다는, 다음 흐름으로 공부한 내용을 정리한 저장소에 가깝습니다.

`모델 구조 이해 -> 직접 구현 -> 실제 학습 -> 결과 저장`

- Transformer: 구현 중심
- DDPM: TinyHero 기반 이미지 생성 학습
- Latent Diffusion + VAE: latent space 기반 생성 실험
- Flow Matching: TinyHero 기반 flow matching 학습 및 체크포인트 정리
- Diffusion Policy: imitation/action prediction 실험
