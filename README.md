# ai_architecture_study

AI 모델 구조 공부하면서 이것저것 직접 구현해본 저장소임.  
크게 `transformer`, `ddpm`, `diffusion_policy` 세 파트로 나눠서 정리해둠. 어떤 건 구현만 해뒀고, 어떤 건 학습 결과 이미지랑 weight까지 같이 저장해둠.

## 전체적으로 보면

- `pyproject.toml`
  - 프로젝트에서 쓰는 패키지 적어둔 파일임.
  - `torch`, `torchvision`, `numpy`, `gymnasium[mujoco]` 같은 것들 들어감.

- `uv.lock`
  - 패키지 버전 고정용 파일임.

- `main.py`
  - 지금은 그냥 `"Hello from ai-architecture!"`만 출력함.
  - 실제 실험 코드는 아래 폴더들에 넣어둠.

## 1. transformer

여기는 Transformer 직접 구현한 폴더임.

- `transformer/model.py`
  - Transformer 기본 부품들 직접 구현한 파일임.
  - embedding, positional encoding, multi-head attention, encoder/decoder block 같은 것들 들어감.
  - 라이브러리 호출해서 바로 쓰는 게 아니라 구조 이해하려고 하나씩 구현한 코드임.

- `transformer/train.ipynb`
  - Transformer 학습용 노트북임. 아직 결과까지 정리해두진 않았음.
  - `Config`, `train()` 같은 형태로 되어 있고 데이터셋 불러와서 학습하는 구조임.

- `transformer/tokenizer_english.json`
  - 영어 토크나이저 파일임.
  - vocab 크기는 `30000`으로 맞춰둠.

- `transformer/tokenizer_korean.json`
  - 한국어 토크나이저 파일임.
  - 이것도 vocab 크기 `30000`으로 맞춰둠.

### 결과

Transformer 쪽은 일단 "구현해둔 상태"에 가까움.  
토크나이저 파일이랑 모델 코드는 정리해뒀는데, 학습 결과물이나 체크포인트까지는 아직 같이 안 올려둠.

## 2. ddpm

여기는 이미지 생성 쪽 실험임.  
TinyHero 데이터셋 가지고 diffusion 모델 학습해본 내용 정리해둔 폴더임.

- `ddpm/ddpm_model.py`
  - DDPM에서 쓰는 U-Net 비슷한 모델 정의 파일임.
  - residual block, down/up block, time embedding 같은 것들 들어감.

- `ddpm/ddpm_train.ipynb`
  - 기본 DDPM 학습 노트북임.
  - 설정값 보면 `64x64` 이미지, `timesteps=500`, `batch_size=100`, `epoch=300` 정도로 학습하게 되어 있음.
  - 매 epoch마다 샘플 이미지 저장하고 마지막에 weight 저장하는 식임.

- `ddpm/tinyhero.zip`
  - TinyHero 데이터 압축 파일임.

- `ddpm/datas/tinyhero/`
  - 실제 학습에 쓰는 이미지들이 들어 있는 폴더임.
  - PNG 이미지 총 `3648`장으로 구성해둠.

- `ddpm/ddpm_img/`
  - 학습 중간중간 생성한 샘플 이미지 저장된 폴더임.

- `ddpm/weight/ddpm.pth`
  - 마지막에 저장한 DDPM weight임.

### 결과

이쪽은 실제로 학습 돌린 결과까지 같이 넣어둠.

- 노트북 출력에는 `Epoch 1 / 300` 로그가 남도록 해뒀음.
- 첫 epoch 쪽 로그에서는 `Loss=0.119` 나왔음.
- 마지막에 `Training Complete`도 찍히게 해둠.
- `ddpm/ddpm_img` 폴더에는 샘플 이미지 `299`장 저장해둠.
  - 파일명은 `ddpm_epoch_001.png`부터 `ddpm_epoch_299.png`까지임.
- 최종 weight는 `ddpm/weight/ddpm.pth`로 저장해둠.

즉 DDPM은 코드만 적어둔 게 아니라 실제로 학습하고 결과까지 같이 저장해둠.

## 3. latent diffusion + vae

같은 `ddpm` 폴더 안에 latent diffusion 실험도 같이 넣어뒀음.  
먼저 VAE 학습해서 latent space 만들고, 그 위에서 diffusion 학습하는 방식으로 해봤음.

- `ddpm/ldpm_model.py`
  - VAE 모델 정의 파일임.
  - encoder, decoder, reparameterize, KL loss 들어감.

- `ddpm/ldpm_train.ipynb`
  - VAE 학습하고 그다음 latent diffusion 학습하는 노트북임.
  - 설정 보면 `vae_n_epoch=2000`, `n_epoch=7000`, `latent_dim=128`로 되어 있음.
  - VAE는 60 epoch마다, latent diffusion은 100 epoch마다 결과 저장하게 되어 있음.

- `ddpm/vae_img/`
  - VAE reconstruction 결과 이미지 폴더임.

- `ddpm/ldpm_img/`
  - latent diffusion 샘플 이미지 폴더임.

- `ddpm/ldpm_weight/`
  - VAE랑 latent diffusion weight 같이 저장되는 폴더임.

### 결과

이 부분도 중간 결과랑 weight를 꽤 많이 저장해둠.

- VAE 쪽 첫 epoch 로그에는
  - `loss=0.2356`
  - `recon=0.2354`
  - `kl=0.1839`
  - `beta=0.001`
  이런 값들이 찍혔음.

- VAE 결과는 `ddpm/vae_img`에 총 `33`장 저장해둠.
  - 파일명은 `vae_epoch_060.png`부터 `vae_epoch_1980.png`까지임.

- latent diffusion 쪽은 첫 epoch 로그에 `Loss=1.02` 나왔음.

- `ddpm/ldpm_img`에는 샘플 이미지 총 `13`장 저장해둠.
  - `ldm_epoch_100.png`부터 `ldm_epoch_1300.png`까지 저장함.

- `ddpm/ldpm_weight` 폴더에는
  - `vae.pth`
  - `vae_60epoch.pth` 같은 중간 VAE weight들
  - `ldm_100epoch.pth`부터 `ldm_1300epoch.pth`까지의 diffusion weight들
  이 같이 넣어둠.

정리하면 이 파트는 "VAE 먼저 학습 -> latent diffusion 학습" 흐름으로 진행했고, 결과물도 같이 저장해둠.

## 4. diffusion_policy

여기는 이미지 생성이 아니라 행동(action) 예측 쪽 실험임.  
관측값 보고 앞으로의 action chunk 예측하는 diffusion policy 구조를 구현해본 내용임.

- `diffusion_policy/based_cnn_model.py`
  - diffusion policy 모델 정의 파일임.
  - 1D convolution 들어가고 timestep embedding이랑 observation conditioning 같이 씀.

- `diffusion_policy/process_data.py`
  - 원본 데이터를 학습용 형태로 바꾸는 전처리 코드임.
  - 각 시점 observation에 대해 앞으로 `horizon`만큼의 action 잘라서 `action chunk` 만드는 방식임.
  - 길이 모자라면 마지막 action으로 padding함.

- `diffusion_policy/base_cnn_train.ipynb`
  - 전처리된 데이터 가지고 실제 학습하는 노트북임.
  - 데이터 로드, dataset 정의, 모델 생성, 학습, 샘플링까지 한 번에 들어 있음.

- `diffusion_policy/datas/reach_bc.npz`
  - 원본 rollout 데이터임.

- `diffusion_policy/datas/reach_bc_imitation_h15.npz`
  - 전처리 후 만들어진 데이터임.
  - 이름 그대로 `horizon=15` 기준임.

### 결과

이쪽도 전처리 결과랑 학습 로그를 같이 남겨둠.

- 원본 `reach_bc.npz`는 episode 단위 object 배열 형태로 저장돼 있음.

- 전처리 후 데이터 `reach_bc_imitation_h15.npz`는 shape을 다음처럼 맞춰둠.
  - `observations`: `(1000000, 32)`
  - `action_chunks`: `(1000000, 15, 7)`
  - `action_chunks_t`: `(1000000, 7, 15)`

즉 observation 차원은 `32`, action 차원은 `7`, horizon은 `15`임.

- 학습 노트북 출력에는
  - 데이터셋 크기 `1,000,000`
  - 배치 shape `torch.Size([64, 32])`, `torch.Size([64, 7, 15])`
  - 모델 파라미터 수 `207367`
  이렇게 나오게 해둠.

- 학습 로그에는 `epoch=001 mean_loss=1.302935`가 찍혔음.

- 마지막 쪽에는 `ground truth chunk[0]`, `sampled chunk[0]` 출력도 넣어둬서 샘플링 결과까지 같이 볼 수 있게 해둠.

## 한 줄로 정리하면

이 저장소는 그냥 예제만 모아둔 게 아니라,  
"모델 구조 공부 -> 직접 구현 -> 실제로 조금씩 학습해봄 -> 결과 저장"  
이 흐름대로 공부한 내용 정리해둔 저장소임.

- Transformer: 구현 위주로 정리해둠
- DDPM: TinyHero로 학습하고 결과 저장해둠
- Latent Diffusion + VAE: 중간 결과 이미지랑 weight 같이 저장해둠
- Diffusion Policy: 전처리 결과랑 학습 로그까지 같이 정리해둠
