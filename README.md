# ai_architecture_study

AI 아키텍처를 공부하면서 직접 구현하고, 일부는 실제 학습 결과까지 정리한 저장소입니다.  
현재는 `transformer`, `ddpm`, `flow_matching`, `diffusion_transformer`, `diffusion_policy`를 중심으로 구성되어 있습니다.

## Overview

- 목적: 모델 구조 이해, 직접 구현, 학습 실험, 결과 정리
- 주요 라이브러리: `torch`, `torchvision`, `numpy`, `tqdm`, `gymnasium[mujoco]`
- 의존성 관리: `pyproject.toml`, `uv.lock`

## Directory Guide

### `transformer/`

Transformer 기본 구조를 직접 구현한 파트입니다.

- `model.py`
  - embedding, positional encoding, multi-head attention, encoder/decoder block 구현
- `train.ipynb`
  - Transformer 학습용 노트북
- `tokenizer_english.json`, `tokenizer_korean.json`
  - 영어/한국어 토크나이저

상태:
구현 중심이며, 학습 결과물보다는 모델 구조 정리에 초점이 있습니다.

### `ddpm/`

TinyHero 데이터셋으로 diffusion 계열 모델을 실험한 파트입니다.

- `ddpm_model.py`
  - DDPM용 U-Net 계열 모델
- `ddpm_train.ipynb`
  - 기본 DDPM 학습 노트북
- `ldpm_model.py`
  - VAE 및 latent diffusion 관련 모델
- `ldpm_train.ipynb`
  - VAE + latent diffusion 학습 노트북
- `datas/tinyhero/`
  - TinyHero 학습 데이터
- `ddpm_img/`, `ldpm_img/`
  - 학습 중 저장한 샘플 이미지
- `weight/`, `ldpm_weight/`
  - 학습 weight

상태:
구현뿐 아니라 샘플 이미지와 weight까지 함께 포함되어 있습니다.

### `flow_matching/`

TinyHero 이미지에 대해 flow matching 기반 생성 모델을 학습한 파트입니다.

- `flow_matching_unet_model.py`
  - flow matching용 U-Net 계열 모델
- `flow_matching_train.ipynb`
  - 학습 및 샘플링 노트북

핵심 아이디어:
`x_t = t * x + (1 - t) * noise` 형태의 interpolation을 만들고, 모델이 vector field를 예측하도록 학습합니다.

### `diffusion_transformer/`

Diffusion Transformer 기반 생성 모델을 실험한 파트입니다.

- `models.py`
  - `DiT` 모델 구현
  - patch embedding, Linformer attention, adaLN-zero 스타일 conditioning 포함
- `dit_train.ipynb`
  - TinyHero 데이터 기반 학습 노트북
- `dit_img/`
  - epoch별 샘플 이미지
- `dit_weight/`
  - 중간 및 최종 weight

상태:
Transformer 기반 diffusion 구조를 별도 디렉토리로 정리했고, 학습 결과도 함께 저장되어 있습니다.

### `diffusion_policy/`

이미지 생성이 아니라 observation으로부터 미래 action chunk를 예측하는 diffusion policy 실험입니다.

- `based_cnn_model.py`
  - 1D CNN 기반 diffusion policy 모델
- `process_data.py`
  - rollout 데이터를 학습용 action chunk 형태로 변환하는 전처리 코드
- `base_cnn_train.ipynb`
  - 기본 학습 노트북
- `base_cnn_train_normalized.ipynb`
  - 정규화 버전 실험 노트북
- `show_data.ipynb`
  - 데이터 확인용 노트북

상태:
전처리, 학습, 샘플링 흐름이 모두 포함된 imitation learning 실험입니다.

## Project Structure

```text
ai_architecture_study/
├── transformer/
├── ddpm/
├── flow_matching/
├── diffusion_transformer/
├── diffusion_policy/
├── pyproject.toml
├── uv.lock
└── README.md
```

## How To Run

대부분의 실험은 각 디렉토리의 `ipynb` 노트북 기준으로 정리되어 있습니다.

- Transformer: `transformer/train.ipynb`
- DDPM: `ddpm/ddpm_train.ipynb`
- Latent Diffusion: `ddpm/ldpm_train.ipynb`
- Flow Matching: `flow_matching/flow_matching_train.ipynb`
- Diffusion Transformer: `diffusion_transformer/dit_train.ipynb`
- Diffusion Policy: `diffusion_policy/base_cnn_train.ipynb`

노트북마다 데이터 경로가 상대 경로로 연결되어 있으니, 실행 위치와 데이터 디렉토리를 먼저 확인하는 것이 좋습니다.

## Summary

이 저장소는 다음 흐름으로 정리되어 있습니다.

`모델 구조 이해 -> 직접 구현 -> 학습 실험 -> 결과 저장`

- `transformer`: 구조 구현 중심
- `ddpm`: 이미지 생성 diffusion 실험
- `flow_matching`: flow 기반 생성 실험
- `diffusion_transformer`: Transformer 기반 diffusion 실험
- `diffusion_policy`: action prediction 기반 diffusion policy 실험
