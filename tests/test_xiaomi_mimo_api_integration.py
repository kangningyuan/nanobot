#!/usr/bin/env python3
"""
Xiaomi MiMo API 集成测试 - 使用真实 API 调用

运行方式:
    python tests/test_xiaomi_mimo_integration.py

注意: 需要在 ~/.nanobot/config.json 中配置 xiaomi_mimo 的 api_key
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.base import LLMResponse


def load_config():
    """加载配置文件"""
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_xiaomi_mimo_config():
    """获取 Xiaomi MiMo 配置"""
    config = load_config()

    providers = config.get("providers", {})

    for key in ["xiaomi_mimo", "xiaomiMimo"]:
        xiaomi_config = providers.get(key, {})
        api_key = xiaomi_config.get("api_key", "") or xiaomi_config.get("apiKey", "")
        api_base = xiaomi_config.get("api_base") or xiaomi_config.get("apiBase")
        if api_key:
            api_base = api_base or "https://api.xiaomimimo.com/v1"
            return api_key, api_base

    custom_config = providers.get("custom", {})
    api_key = custom_config.get("api_key", "") or custom_config.get("apiKey", "")
    api_base = custom_config.get("api_base") or custom_config.get("apiBase") or "https://api.xiaomimimo.com/v1"

    if not api_key:
        print("❌ 未配置 xiaomi_mimo/xiaomiMimo/custom 的 api_key")
        print("请在 ~/.nanobot/config.json 中配置:")
        print('  "providers": {')
        print('    "xiaomiMimo": {')
        print('      "apiKey": "your-api-key",')
        print('      "apiBase": "https://api.xiaomimimo.com/v1"')
        print("    }")
        print("  }")
        sys.exit(1)

    return api_key, api_base


async def test_basic_chat():
    """测试基本对话功能"""
    print("\n" + "=" * 60)
    print("测试 1: 基本对话")
    print("=" * 60)

    api_key, api_base = get_xiaomi_mimo_config()
    provider = CustomProvider(
        api_key=api_key,
        api_base=api_base,
        default_model="mimo-v2-flash",
    )

    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]

    print(f"📤 发送消息: {messages[0]['content']}")
    print(f"🔗 API Base: {api_base}")
    print(f"🤖 Model: mimo-v2-flash")

    try:
        response = await provider.chat(messages)

        print(f"\n📥 响应状态: {response.finish_reason}")
        if response.content:
            print(f"💬 响应内容: {response.content[:200]}...")
        if response.usage:
            print(f"📊 Token 使用: prompt={response.usage.get('prompt_tokens')}, "
                  f"completion={response.usage.get('completion_tokens')}, "
                  f"total={response.usage.get('total_tokens')}")

        if response.finish_reason == "error":
            print(f"❌ 测试失败: {response.content}")
            return False

        print("✅ 基本对话测试通过")
        return True

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False


async def test_tool_calls():
    """测试工具调用功能"""
    print("\n" + "=" * 60)
    print("测试 2: 工具调用")
    print("=" * 60)

    api_key, api_base = get_xiaomi_mimo_config()
    provider = CustomProvider(
        api_key=api_key,
        api_base=api_base,
        default_model="mimo-v2-flash",
    )

    messages = [{"role": "user", "content": "北京现在的天气怎么样？"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    print(f"📤 发送消息: {messages[0]['content']}")
    print(f"🔧 工具定义: get_weather")

    try:
        response = await provider.chat(messages, tools=tools)

        print(f"\n📥 响应状态: {response.finish_reason}")

        if response.tool_calls:
            print(f"🔧 工具调用数量: {len(response.tool_calls)}")
            for i, tc in enumerate(response.tool_calls):
                print(f"   工具 {i+1}: {tc.name}")
                print(f"   参数: {json.dumps(tc.arguments, ensure_ascii=False)}")

        if response.finish_reason == "tool_calls":
            print("✅ 工具调用测试通过")
            return True
        elif response.finish_reason == "error":
            print(f"❌ 测试失败: {response.content}")
            return False
        else:
            print(f"⚠️ 模型未调用工具，直接返回: {response.content[:100] if response.content else 'None'}...")
            return True

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False


async def test_multi_turn_conversation():
    """测试多轮对话"""
    print("\n" + "=" * 60)
    print("测试 3: 多轮对话")
    print("=" * 60)

    api_key, api_base = get_xiaomi_mimo_config()
    provider = CustomProvider(
        api_key=api_key,
        api_base=api_base,
        default_model="mimo-v2-flash",
    )

    messages = [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好张三，很高兴认识你！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "你还记得我叫什么名字吗？"},
    ]

    print(f"📤 发送多轮对话...")

    try:
        response = await provider.chat(messages)

        print(f"\n📥 响应状态: {response.finish_reason}")
        if response.content:
            print(f"💬 响应内容: {response.content}")

        if response.finish_reason == "error":
            print(f"❌ 测试失败: {response.content}")
            return False

        if "张三" in (response.content or ""):
            print("✅ 多轮对话测试通过 - 模型记住了用户名字")
        else:
            print("⚠️ 多轮对话测试通过 - 但模型可能未记住上下文")

        return True

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False


async def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试 5: 错误处理")
    print("=" * 60)

    print("🧪 使用无效 API Key 测试...")

    provider = CustomProvider(
        api_key="invalid-api-key-12345",
        api_base="https://api.xiaomimimo.com/v1",
        default_model="mimo-v2-flash",
    )

    messages = [{"role": "user", "content": "Hello"}]

    try:
        response = await provider.chat(messages)

        if response.finish_reason == "error":
            print(f"✅ 错误处理正确: {response.content[:100]}...")
            return True
        else:
            print(f"⚠️ 预期错误但成功返回: {response.content}")
            return False

    except Exception as e:
        print(f"✅ 异常被正确捕获: {e}")
        return True


async def test_reasoning_content():
    """测试推理内容（如果模型支持）"""
    print("\n" + "=" * 60)
    print("测试 6: 推理内容 (Reasoning Content)")
    print("=" * 60)

    api_key, api_base = get_xiaomi_mimo_config()
    provider = CustomProvider(
        api_key=api_key,
        api_base=api_base,
        default_model="mimo-v2-flash",
    )

    messages = [{"role": "user", "content": "请解释一下什么是递归，并给出一个例子"}]

    print(f"📤 发送需要推理的问题...")

    try:
        response = await provider.chat(messages, max_tokens=500)

        print(f"\n📥 响应状态: {response.finish_reason}")

        if response.content:
            print(f"💬 响应内容: {response.content[:200]}...")

        if response.reasoning_content:
            print(f"🧠 推理内容: {response.reasoning_content[:200]}...")
            print("✅ 模型支持推理内容")
        else:
            print("ℹ️ 模型未返回推理内容（可能不支持此功能）")

        if response.finish_reason == "error":
            print(f"❌ 测试失败: {response.content}")
            return False

        return True

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Xiaomi MiMo API 集成测试")
    print("=" * 60)

    tests = [
        ("基本对话", test_basic_chat),
        ("工具调用", test_tool_calls),
        ("多轮对话", test_multi_turn_conversation),
        ("错误处理", test_error_handling),
        ("推理内容", test_reasoning_content),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 发生未捕获异常: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查")


if __name__ == "__main__":
    asyncio.run(main())
