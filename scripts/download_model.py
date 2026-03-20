#!/usr/bin/env python3
"""
Скрипт для загрузки модели Llama 3.1 8B Instruct
"""

import argparse
import sys
from pathlib import Path

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llama_client import LlamaClient


def download_model(quantization: str = "Q5_K_M", force: bool = False):
    """
    Скачать модель
    
    Args:
        quantization: Уровень квантования (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
        force: Перезаписать существующий файл
    """
    client = LlamaClient(quantization=quantization)
    model_path = client.get_model_path()
    model_info = client.get_model_info()
    
    print(f"\n🦙 Llama 3.1 8B Instruct — Загрузка модели")
    print(f"=" * 50)
    print(f"Квантование: {quantization}")
    print(f"Размер: ~{model_info['size_gb']} GB")
    print(f"Качество: {model_info['quality']}")
    print(f"Путь: {model_path}")
    print()
    
    if model_path.exists() and not force:
        print(f"✅ Модель уже существует: {model_path.name}")
        print(f"   Используйте --force для перезагрузки")
        return
    
    if model_path.exists() and force:
        print(f"⚠️  Существующий файл будет удалён")
    
    # Создать директорию
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Загрузка модели...")
    print(f"   URL: {model_info['url']}")
    print()
    
    # Использовать wget или requests для загрузки
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(model_info['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc=model_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        print(f"\n✅ Модель успешно загружена: {model_path.name}")
        print(f"   Размер: {model_path.stat().st_size / 1024**3:.2f} GB")
        
    except ImportError:
        print("❌ Ошибка: установите зависимости:")
        print("   pip install requests tqdm")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        sys.exit(1)


def list_models():
    """Показать доступные модели для загрузки"""
    client = LlamaClient()
    
    print(f"\n🦙 Llama 3.1 8B Instruct — Доступные модели")
    print(f"=" * 50)
    print()
    
    for q, info in client.MODELS.items():
        exists = client.get_model_path(q).exists()
        status = "✅" if exists else "⬜"
        print(f"{status} {q:8} | ~{info['size_gb']} GB | {info['quality']:10} | {info['file']}")
    
    print()
    print(f"Рекомендуемая: Q5_K_M (баланс качество/размер)")
    print(f"Для загрузки: python scripts/download_model.py --quantization Q5_K_M")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка модели Llama 3.1 8B Instruct"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="Q5_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Уровень квантования (по умолчанию: Q5_K_M)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Перезаписать существующий файл"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Показать доступные модели"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    else:
        download_model(args.quantization, args.force)


if __name__ == "__main__":
    main()
