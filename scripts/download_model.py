#!/usr/bin/env python3
"""
Универсальный скрипт для скачивания LLM моделей

Поддерживает все модели из реестра:
- Llama 3.1 8B
- Vikhr-Llama-3.1-8B-Instruct-R
- Vikhr-Nemo-12B-Instruct-R
- Qwen 2.5 7B

Примеры:
    python scripts/download_model.py --model vikhr-nemo-12b-instruct-r
    python scripts/download_model.py --model llama-3.1-8b-instruct --quantization Q4_K_M
    python scripts/download_model.py --list
"""

import argparse
import os
import sys
from pathlib import Path

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.registry import (
    list_models,
    get_model,
    get_model_file_path,
    get_model_url,
    get_model_size,
    get_recommended_quantization,
)
from src.utils.config import Config


def download_model(model_id: str, quantization: str, force: bool = False):
    """
    Скачать модель
    
    Args:
        model_id: ID модели из реестра
        quantization: Уровень квантования
        force: Перезаписать существующий файл
    """
    # Получить информацию о модели
    model_info = get_model(model_id)
    if not model_info:
        print(f"❌ Модель '{model_id}' не найдена в реестре")
        print(f"\nДоступные модели:")
        list_models_cmd()
        sys.exit(1)
    
    # Получить путь и URL
    config = Config.load()
    filename = get_model_file_path(model_id, quantization)
    model_path = config.models_dir_path / filename
    model_url = get_model_url(model_id, quantization)
    model_size = get_model_size(model_id, quantization)
    
    print(f"\n📥 Загрузка модели")
    print(f"=" * 60)
    print(f"Модель: {model_info['display_name']}")
    print(f"ID: {model_id}")
    print(f"Квантование: {quantization}")
    print(f"Размер: ~{model_size} GB")
    print(f"Контекст: {model_info['context_window']} токенов")
    print(f"Путь: {model_path}")
    print(f"URL: {model_url[:80]}...")
    print()
    
    # Проверка существования
    if model_path.exists() and not force:
        print(f"✅ Модель уже существует: {model_path.name}")
        print(f"   Используйте --force для перезагрузки")
        return
    
    if model_path.exists() and force:
        print(f"⚠️  Существующий файл будет удалён")
        model_path.unlink()
    
    # Создать директорию
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Загрузка началась...")
    print()
    
    # Загрузка с поддержкой HF_TOKEN
    try:
        import requests
        from tqdm import tqdm
        
        # Заголовки для авторизации
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        response = requests.get(model_url, stream=True, headers=headers)
        
        # Проверка на ошибку авторизации
        if response.status_code == 401:
            print("\n❌ Ошибка авторизации (HTTP 401)")
            print("   Модель требует аутентификации на Hugging Face")
            print("   Установите HF_TOKEN в .env или используйте модель без аутентификации")
            print()
            print("   Пример .env:")
            print("   HF_TOKEN=your_token_here")
            print()
            if model_path.exists():
                model_path.unlink()
            sys.exit(1)
        
        # Проверка на другие ошибки
        if response.status_code != 200:
            print(f"\n❌ Ошибка загрузки (HTTP {response.status_code})")
            print(f"   URL: {model_url}")
            if model_path.exists():
                model_path.unlink()
            sys.exit(1)
        
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            print("❌ Ошибка: не удалось получить размер файла")
            if model_path.exists():
                model_path.unlink()
            sys.exit(1)
        
        with open(model_path, 'wb') as f, tqdm(
            desc=model_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        # Проверка размера
        actual_size = model_path.stat().st_size
        expected_size = model_size * 1024**3
        size_diff = abs(actual_size - expected_size) / expected_size * 100
        
        print(f"\n✅ Модель успешно загружена: {model_path.name}")
        print(f"   Размер: {actual_size / 1024**3:.2f} GB")
        
        if size_diff > 10:
            print(f"   ⚠️  Размер отличается от ожидаемого на {size_diff:.1f}%")
            print(f"   Возможно, файл повреждён. Попробуйте скачать повторно.")
        
    except ImportError:
        print("❌ Ошибка: установите зависимости:")
        print("   pip install requests tqdm")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        if model_path.exists():
            model_path.unlink()
        sys.exit(1)


def list_models_cmd():
    """Показать доступные модели"""
    config = Config.load()
    
    print(f"\n📚 Доступные модели для загрузки")
    print(f"=" * 80)
    print()
    
    models = list_models()
    
    for i, model_id in enumerate(models, 1):
        model_info = get_model(model_id)
        if not model_info:
            continue
        
        recommended_q = get_recommended_quantization(model_id)
        
        print(f"{i}. {model_info['display_name']}")
        print(f"   ID: {model_id}")
        print(f"   Контекст: {model_info['context_window']:,} токенов")
        print(f"   Рекомендуемое квантование: {recommended_q}")
        print()
        
        # Размеры для каждого квантования
        print(f"   Доступные варианты:")
        for q, size in model_info["sizes"].items():
            filename = get_model_file_path(model_id, q)
            model_path = config.models_dir_path / filename
            exists = model_path.exists()
            status = "✅" if exists else "⬜"
            print(f"     {status} {q:8} | ~{size} GB | {filename}")
        
        print()
        print(f"   {model_info['description']}")
        print()
        print(f"   " + "-" * 70)
        print()
    
    print(f"Примеры использования:")
    print(f"  python scripts/download_model.py --model llama-3.1-8b-instruct")
    print(f"  python scripts/download_model.py --model vikhr-nemo-12b-instruct-r")
    print(f"  python scripts/download_model.py --model qwen-2.5-7b-instruct -q Q4_K_M")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Скачивание LLM моделей для SmartTherm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s                          # Скачать дефолтную модель (из configs/default.yaml)
  %(prog)s --model vikhr-nemo-12b-instruct-r
  %(prog)s --model llama-3.1-8b-instruct --quantization Q4_K_M
  %(prog)s --list
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="ID модели из реестра (по умолчанию: из configs/default.yaml)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default=None,
        help="Уровень квантования (по умолчанию: recommended для модели или из конфига)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Перезаписать существующий файл"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Показать все доступные модели"
    )

    args = parser.parse_args()

    if args.list:
        list_models_cmd()
    else:
        # Загрузить конфиг для получения дефолтной модели
        config = Config.load()
        
        # Модель: аргумент или дефолтная из конфига
        model_id = args.model or config.llm.get("model")
        
        if not model_id:
            print("❌ Модель не указана")
            print()
            print("Укажите --model или настройте llm.model в configs/default.yaml")
            print()
            print("Примеры:")
            print("  python scripts/download_model.py --model vikhr-nemo-12b-instruct-r")
            print("  python scripts/download_model.py --list")
            sys.exit(1)
        
        # Квантование: аргумент или recommended для модели
        quantization = args.quantization or get_recommended_quantization(model_id)
        
        download_model(model_id, quantization, args.force)


if __name__ == "__main__":
    main()
