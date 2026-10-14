"""A terminal answer replaces DOM progress fragments, with exact receipt hash."""
from types import SimpleNamespace
import hashlib
import time
from chatgpt_web2api.turn_anchor import TurnAnchor
from chatgpt_web2api.provider_contract import summarize_turn_anchor
import json
import pytest
from chatgpt_web2api.api_server import APIServer
from chatgpt_web2api.cdp_driver import StreamChunk
from chatgpt_web2api.completion_detector import DetectorBudgets
from chatgpt_web2api.provider_contract import ProviderOperationContext, ProviderOperationReceipt, EffectCertainty

@pytest.mark.asyncio
@pytest.mark.parametrize("progress,final", [
    ("I will inspect. A very long progress update that is not the answer.", "Done."),
    ("Beginning... fragment middle corrupted suffix", "The source change is verified and the existing service is running."),
    ("obsolete", "correct!"),
])
async def test_strict_full_response_uses_authoritative_terminal_text(monkeypatch,progress,final):
    monkeypatch.setattr(DetectorBudgets,"from_config",classmethod(lambda cls,*args:cls.default()))
    context=ProviderOperationContext(operation_id="test-terminal-answer",requested_project_id="project-a",observed_project_id="project-a",requested_model="gpt-6",observed_model="gpt-6")
    receipt=ProviderOperationReceipt(operation_id=context.operation_id,conversation_id="conversation-a",target_id="target-a",session_identity="fixture",captured_user_message_id="user-a",turn_anchor=summarize_turn_anchor(TurnAnchor(sent_text="original prompt",mode="fresh_chat",pre_send_wall_time=time.time(),conversation_id_at_capture=None).with_captured_id("user-a")),requested_project_id="project-a",observed_project_id="project-a",requested_model="gpt-6",observed_model="gpt-6",response_status="COMPLETE",effect_certainty=EffectCertainty.CONFIRMED_COMPLETE,response_sha256=hashlib.sha256(final.encode()).hexdigest())
    calls=[]
    async def send(text,**kwargs):
        calls.append(text)
        yield StreamChunk(delta=progress)
        yield StreamChunk(delta="",finish_reason="stop",provider_receipt=receipt.to_dict(),final_text=final)
    server=APIServer.__new__(APIServer)
    server._config=SimpleNamespace(chatgpt=object())
    server._driver=SimpleNamespace(send_and_stream=send)
    result=await server._full_response(None,"gpt-6","original prompt",120,operation_context=context)
    body=json.loads(result.body)
    assert body["choices"][0]["message"]["content"]==final
    assert body["conversation_id"]=="conversation-a"
    assert len(calls)==1

@pytest.mark.asyncio
async def test_terminal_hash_mismatch_cannot_report_success(monkeypatch):
    monkeypatch.setattr(DetectorBudgets,"from_config",classmethod(lambda cls,*args:cls.default()))
    context=ProviderOperationContext(operation_id="test-terminal-mismatch",requested_project_id="project-a",observed_project_id="project-a",requested_model="gpt-6",observed_model="gpt-6")
    receipt=ProviderOperationReceipt(operation_id=context.operation_id,conversation_id="conversation-a",target_id="target-a",session_identity="fixture",captured_user_message_id="user-a",turn_anchor=summarize_turn_anchor(TurnAnchor(sent_text="original prompt",mode="fresh_chat",pre_send_wall_time=time.time(),conversation_id_at_capture=None).with_captured_id("user-a")),requested_project_id="project-a",observed_project_id="project-a",requested_model="gpt-6",observed_model="gpt-6",response_status="COMPLETE",effect_certainty=EffectCertainty.CONFIRMED_COMPLETE,response_sha256=hashlib.sha256(b"different answer").hexdigest())
    async def send(text,**kwargs):
        yield StreamChunk(delta="old progress")
        yield StreamChunk(delta="",finish_reason="stop",provider_receipt=receipt.to_dict(),final_text="unverified answer")
    server=APIServer.__new__(APIServer);server._config=SimpleNamespace(chatgpt=object());server._driver=SimpleNamespace(send_and_stream=send)
    with pytest.raises(RuntimeError,match="terminal response hash mismatch"):
        await server._full_response(None,"gpt-6","original prompt",120,operation_context=context)
