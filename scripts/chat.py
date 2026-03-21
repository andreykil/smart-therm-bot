#!/usr/bin/env python3
"""
Универсальный интерактивный чат с LLM

Поддерживает все модели из реестра:
- Llama 3.1 8B
- Vikhr-Llama-3.1-8B-Instruct-R
- Vikhr-Nemo-12B-Instruct-R
- Qwen 2.5 7B

Примеры:
    python scripts/chat.py --model vikhr-nemo-12b-instruct-r
    python scripts/chat.py --model llama-3.1-8b-instruct --context 16384
    python scripts/chat.py --prompt "Как подключить WiFi?" --model vikhr-llama-3.1-8b-instruct-r
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import create_llm_engine, LLMEngine
from src.llm.registry import get_model, get_recommended_quantization
from src.utils.chat_format import format_llama_chat_prompt, clean_response_text


class ChatSession:
    """Сессия чата с моделью"""
    
    def __init__(self, client: LLMEngine, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.history = []
    
    def send(self, user_message: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Отправить сообщение и получить ответ"""
        self.messages.append({"role": "user", "content": user_message})

        # Форматирование промпта для Llama 3.1
        prompt = format_llama_chat_prompt(self.messages)

        response = self.client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>"]
        )

        # Очистка от специальных токенов (полный ответ, strip=True)
        response = clean_response_text(response, strip_spaces=True)

        self.messages.append({"role": "assistant", "content": response})
        self.history.append({"user": user_message, "assistant": response})

        return response
    
    def clear(self):
        """Очистить историю чата"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.history = []
    
    def save(self, filepath: str):
        """Сохранить историю чата"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"💾 История сохранена: {filepath}")


def interactive_chat(client: LLMEngine, system_prompt: str, max_tokens: int = 1024):
    """Интерактивный режим чата"""
    session = ChatSession(client, system_prompt)
    
    model_info = get_model(client.model_id)
    
    print("\n" + "=" * 70)
    print(f"💬 {model_info['display_name'] if model_info else client.model_id} — Интерактивный чат")
    print("=" * 70)
    print(f"Системный промпт: {system_prompt[:70]}...")
    print()
    print("Команды:")
    print("  /clear     — очистить историю")
    print("  /save      — сохранить историю")
    print("  /stats     — показать статистику")
    print("  /exit      — выйти")
    print("=" * 70)
    print()
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "/exit":
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == "/clear":
                session.clear()
                print("🗑️  История очищена")
                continue
            
            if user_input.lower() == "/save":
                filepath = input("Путь для сохранения [chat_history.json]: ").strip()
                if not filepath:
                    filepath = "chat_history.json"
                session.save(filepath)
                continue
            
            if user_input.lower() == "/stats":
                stats = client.get_stats()
                print("\n📊 Статистика:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            # Генерация ответа
            print("\n🤖 Модель: ", end="", flush=True)

            # Форматирование промпта для Llama 3.1
            prompt = format_llama_chat_prompt(session.messages)

            # Используем generate_stream для потокового вывода
            response = ""
            first_token = True
            try:
                for token in session.client.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stop=["<|eot_id|>"]
                ):
                    # Очистка от специальных токенов (только первый токен)
                    if first_token:
                        token = clean_response_text(token, strip_spaces=True)
                        first_token = False
                    else:
                        token = clean_response_text(token, strip_spaces=False)
                    
                    if token:
                        print(token, end="", flush=True)
                        response += token
            except Exception as e:
                print(f"\n❌ Ошибка генерации: {e}")
                session.messages.pop()
                continue

            session.messages.append({"role": "assistant", "content": response})
            session.history.append({"user": user_input, "assistant": response})
            
            print()
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано. Для выхода используйте /exit")
            continue
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            continue


def single_prompt_mode(client: LLMEngine, prompt: str, output_file: Optional[str] = None, max_tokens: int = 1024):
    """Одиночный промпт (не интерактивный)"""
    response = client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    print("\n" + "=" * 70)
    print("📝 Запрос:")
    print(prompt)
    print()
    print("🤖 Ответ:")
    print(response)
    print("=" * 70)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Запрос:\n{prompt}\n\nОтвет:\n{response}\n")
        print(f"💾 Сохранено: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивный чат с LLM моделями",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --model vikhr-nemo-12b-instruct-r
  %(prog)s --model llama-3.1-8b-instruct --context 16384
  %(prog)s --model vikhr-llama-3.1-8b-instruct-r --prompt "Как прошить?"
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=False,
        help="ID модели (например, vikhr-nemo-12b-instruct-r)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        help="Уровень квантования (по умолчанию recommended)"
    )
    parser.add_argument(
        "--context", "-c",
        type=int,
        default=8192,
        help="Размер контекста (по умолчанию: 8192)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=1024,
        help="Максимум токенов на ответ (по умолчанию: 1024)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Одиночный промпт (не интерактивный режим)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Файл для сохранения ответа"
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default="Ты — полезный ассистент SmartTherm. Отвечай на вопросы по контроллерам SmartTherm, OpenTherm, ESP8266/ESP32. Отвечай подробно, технически грамотно, на русском языке.",
        help="Системный промпт"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Включить подробное логирование"
    )
    
    args = parser.parse_args()

    # Загрузить конфиг для получения дефолтной модели
    from src.utils.config import Config
    config = Config.load()
    llm_config = config.llm
    
    # Определить модель (аргумент или дефолтная из конфига)
    # None означает использование модели из конфига
    model_id = args.model or llm_config.get("model") or "vikhr-nemo-12b-instruct-r"
    quantization = args.quantization or llm_config.get("quantization") or "Q8_0"
    
    from src.llm.registry import get_model_file_path
    filename = get_model_file_path(model_id, quantization)
    model_path = config.models_dir_path / filename
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        print()
        print("Скачайте модель:")
        print(f"   python scripts/download_model.py --model {model_id}")
        print()
        print("Или выберите другую модель:")
        print("   python scripts/download_model.py --list")
        sys.exit(1)
    
    # Создание движка
    print(f"🔄 Загрузка модели {model_id} ({quantization})...")
    
    try:
        client = create_llm_engine(
            provider="local",           # type: ignore
            model_id=model_id,          # type: ignore
            quantization=quantization,  # type: ignore
            n_ctx=args.context,         # type: ignore
            verbose=args.verbose
        )
        client.load()
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        sys.exit(1)
    
    print(f"✅ Модель загружена")
    print()
    
    # Режим работы
    if args.prompt:
        single_prompt_mode(client, args.prompt, args.output, args.max_tokens)
    else:
        interactive_chat(client, args.system, args.max_tokens)


if __name__ == "__main__":
    main()
