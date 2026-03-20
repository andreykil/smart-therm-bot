#!/usr/bin/env python3
"""
Интерактивный чат с Llama 3.1 8B Instruct
Тестирование модели на Apple Silicon (M2 Max)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Добавить src в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llama_client import LlamaClient


class ChatSession:
    """Сессия чата с моделью"""
    
    def __init__(
        self,
        client: LlamaClient,
        system_prompt: str = "Ты — полезный ассистент SmartTherm. Отвечай на вопросы по контроллерам SmartTherm, OpenTherm, ESP8266/ESP32. Отвечай подробно, технически грамотно, на русском языке."
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.history = []
    
    def send(self, user_message: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Отправить сообщение и получить ответ"""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat(
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
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


def interactive_chat(client: LlamaClient, system_prompt: str, max_tokens: int = 1024):
    """Интерактивный режим чата"""
    session = ChatSession(client, system_prompt)
    
    print("\n" + "=" * 60)
    print("🦙 Llama 3.1 8B Instruct — Интерактивный чат")
    print("=" * 60)
    print(f"Системный промпт: {system_prompt[:80]}...")
    print()
    print("Команды:")
    print("  /clear     — очистить историю")
    print("  /save      — сохранить историю")
    print("  /stats     — показать статистику")
    print("  /exit      — выйти")
    print("=" * 60)
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
            print("\n🤖 Llama: ", end="", flush=True)
            
            # Добавляем сообщение пользователя в историю
            session.messages.append({"role": "user", "content": user_input})
            
            # Используем chat_stream для потокового вывода
            response = ""
            try:
                for token in session.client.chat_stream(
                    messages=session.messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                ):
                    print(token, end="", flush=True)
                    response += token
            except Exception as e:
                print(f"\n❌ Ошибка генерации: {e}")
                # Убираем последнее сообщение при ошибке
                session.messages.pop()
                continue
            
            # Добавляем ответ ассистента в историю
            # Очистить ответ от <|begin_of_text|> и служебных токенов
            response = response.strip()
            if response.startswith("<|begin_of_text|>"):
                response = response[len("<|begin_of_text|>"):]
            if response.startswith("<|start_header_id|>"):
                # Убрать возможный заголовок ассистента
                response = response.split("<|eot_id|>", 1)[-1].strip() if "<|eot_id|>" in response else response
            
            session.messages.append({"role": "assistant", "content": response})
            session.history.append({"user": user_input, "assistant": response})
            
            print()  # Новая строка после ответа
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано. Для выхода используйте /exit")
            continue
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            continue


def single_prompt(client: LlamaClient, prompt: str, output_file: Optional[str] = None):
    """Одиночный промпт (не интерактивный)"""
    response = client.generate(
        prompt=prompt,
        max_tokens=512,
        temperature=0.7
    )
    
    print("\n" + "=" * 60)
    print("📝 Запрос:")
    print(prompt)
    print()
    print("🤖 Ответ:")
    print(response)
    print("=" * 60)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Запрос:\n{prompt}\n\nОтвет:\n{response}\n")
        print(f"💾 Сохранено: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование Llama 3.1 8B Instruct"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="Q5_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Уровень квантования"
    )
    parser.add_argument(
        "--context", "-c",
        type=int,
        default=8192,
        help="Размер контекста (по умолчанию: 8192)"
    )
    parser.add_argument(
        "--max-tokens", "-m",
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
    
    # Инициализация клиента
    client = LlamaClient(
        quantization=args.quantization,
        n_ctx=args.context,
        verbose=args.verbose
    )
    
    # Проверка наличия модели
    if not client.model_exists():
        print(f"❌ Модель не найдена: {client.get_model_path()}")
        print()
        print("Скачайте модель:")
        print(f"   python scripts/download_model.py --quantization {args.quantization}")
        print()
        print("Или выберите другое квантование:")
        print("   python scripts/download_model.py --list")
        sys.exit(1)
    
    # Загрузка модели
    print(f"🔄 Загрузка модели {args.quantization}...")
    try:
        client.load()
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        sys.exit(1)
    
    print(f"✅ Модель загружена")
    print()
    
    # Режим работы
    if args.prompt:
        single_prompt(client, args.prompt, args.output)
    else:
        interactive_chat(client, args.system, args.max_tokens)


if __name__ == "__main__":
    main()
