# infer.py
"""
简单推理脚本：输入人声 WAV，输出伴奏 + 混音 WAV

用法：
  python infer.py --vocal_path vocal.wav --output_path output.wav --config configs/ar.yaml

可选参数：
  --skip_fine     跳过 fine stage（质量略低，适合还没训练 fine 的情况）
  --seconds N     只取前 N 秒（快速试听）
  --cfg_scale 3.0 CFG 引导强度
"""

import argparse
import logging
import sys
import torch
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_models(config: dict, device: torch.device, skip_fine: bool = False):
    """加载 semantic / coarse / fine 三个 stage 的模型权重"""
    from models.ar_singsong import SingSongAR

    model = SingSongAR(config).to(device)
    ckpt_dir = Path(config["training"]["checkpoint_dir"])

    stages = [
        ("semantic", model.stage_semantic),
        ("coarse",   model.stage_coarse),
    ]
    have_fine = False
    if not skip_fine:
        stages.append(("fine", model.stage_fine))

    for stage_name, stage_model in stages:
        ckpt_path = ckpt_dir / f"ar_{stage_name}" / "ckpt_best.pt"
        if not ckpt_path.exists():
            if stage_name == "fine":
                logger.warning(f"缺少 fine checkpoint，降级为 coarse-only: {ckpt_path}")
                break
            logger.error(f"缺少必要 checkpoint: {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stage_model.load_state_dict(ckpt["model_state"])
        logger.info(f"✓ 加载 {stage_name}  (step={ckpt.get('global_step', '?')})")

        if stage_name == "fine":
            have_fine = True

    model.eval()
    return model, have_fine


def encode_vocal(vocal_path: str, config: dict, device: torch.device, seconds: float = None):
    """加载人声 WAV → HuBERT semantic tokens"""
    from data.retokenize import RetokenizeEncoder
    from utils.audio_utils import load_audio, add_gaussian_noise

    encoder = RetokenizeEncoder(
        kmeans_path=config["codec"]["hubert_kmeans_path"],
        hubert_model=config["codec"]["hubert_model"],
        hubert_layer=config["codec"]["hubert_layer"],
        encodec_bandwidth=config["codec"]["encodec_bandwidth"],
        device=str(device),
    )

    logger.info(f"加载人声: {vocal_path}")
    wav, sr = load_audio(vocal_path, target_sr=config["data"]["sample_rate"])

    # 截断（快速试听用）
    if seconds and seconds > 0:
        wav = wav[..., :int(seconds * config["data"]["sample_rate"])]
        logger.info(f"截取前 {seconds}s")

    wav_noisy = add_gaussian_noise(wav, sigma=0.01)
    tokens = encoder.encode_semantic(wav_noisy).unsqueeze(0).to(device)  # [1, T]
    logger.info(f"Semantic tokens: {tokens.shape[1]} frames ({tokens.shape[1] / 50:.1f}s)")
    return wav, tokens


def generate(model, vocal_semantic, have_fine: bool, config: dict,
             temperature: float, top_k: int, cfg_scale: float):
    """AR 三阶段生成器乐 codes"""
    infer_cfg = config.get("inference", {})

    with torch.no_grad():
        if have_fine:
            # 完整 3-stage 生成
            codes = model.generate(
                vocal_semantic,
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
            )  # [1, 8, T_a]
        else:
            # Coarse-only 降级生成
            T_v = vocal_semantic.shape[1]
            inst_semantic = model.stage_semantic.generate(
                {"vocal_semantic": vocal_semantic},
                num_frames=T_v,
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
            )
            codes = model.stage_coarse.generate(
                {"vocal_semantic": vocal_semantic, "inst_semantic": inst_semantic},
                num_frames=int(T_v * 75 / 50),
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
            )  # [1, 4, T_a]

    logger.info(f"生成 codes: {codes.shape}")
    return codes


def decode_to_wav(codes, have_fine: bool, config: dict, device: torch.device):
    """EnCodec codes → 波形（重采样到 16kHz）"""
    import torchaudio
    from encodec import EncodecModel

    encodec = EncodecModel.encodec_model_24khz()
    bw = float(config["codec"]["encodec_bandwidth"])
    encodec.set_target_bandwidth(bw if have_fine else min(3.0, bw))
    encodec.to(device).eval()

    with torch.no_grad():
        wav_24k = encodec.decode([(codes.to(device), None)])  # [1, 1, T@24kHz]

    wav_24k = wav_24k.squeeze(0).cpu()  # [1, T]
    wav_16k = torchaudio.transforms.Resample(24000, config["data"]["sample_rate"])(wav_24k)
    return wav_16k


def save_outputs(vocal_wav, instr_wav, output_path: str, sample_rate: int):
    """保存伴奏 + 人声伴奏混音"""
    from utils.audio_utils import mix_audio, save_audio

    out = Path(output_path)
    instr_path = str(out).replace(".wav", "_instrumental.wav")
    mixed_path = str(out)

    mixed = mix_audio(vocal_wav, instr_wav)
    save_audio(instr_wav, instr_path, sample_rate)
    save_audio(mixed,    mixed_path,  sample_rate)

    logger.info(f"✓ 伴奏:  {instr_path}")
    logger.info(f"✓ 混音:  {mixed_path}")


def infer(args):
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 推理超参（优先命令行 > 配置文件）
    infer_cfg  = config.get("inference", {})
    temperature = args.temperature or infer_cfg.get("temperature", 0.9)
    top_k       = args.top_k       or infer_cfg.get("top_k",       250)
    cfg_scale   = args.cfg_scale   or infer_cfg.get("cfg_scale",   3.0)

    # 1. 加载模型
    model, have_fine = load_models(config, device, skip_fine=args.skip_fine)

    # 2. 编码人声
    vocal_wav, vocal_semantic = encode_vocal(
        args.vocal_path, config, device, seconds=args.seconds)

    # 3. AR 生成
    logger.info("开始 AR 生成...")
    codes = generate(model, vocal_semantic, have_fine, config,
                     temperature, top_k, cfg_scale)

    # 4. 解码为波形
    logger.info("EnCodec 解码...")
    instr_wav = decode_to_wav(codes, have_fine, config, device)

    # 5. 保存
    save_outputs(vocal_wav, instr_wav, args.output_path,
                 config["data"]["sample_rate"])


def main():
    parser = argparse.ArgumentParser(description="HAFM 推理")
    parser.add_argument("--config",      default="configs/ar.yaml", help="配置文件")
    parser.add_argument("--vocal_path",  required=True,             help="输入人声 .wav")
    parser.add_argument("--output_path", default="output.wav",      help="输出混音 .wav")
    parser.add_argument("--skip_fine",   action="store_true",       help="跳过 fine stage")
    parser.add_argument("--seconds",     type=float, default=None,  help="只取前 N 秒")
    parser.add_argument("--cfg_scale",   type=float, default=None,  help="CFG 引导强度")
    parser.add_argument("--temperature", type=float, default=None,  help="采样温度")
    parser.add_argument("--top_k",       type=int,   default=None,  help="Top-K 采样")
    args = parser.parse_args()

    infer(args)


if __name__ == "__main__":
    main()
