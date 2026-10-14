"""Respect server read cooldown without replaying any provider Send."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from chatgpt_web2api.backend_client import BackendClient
from chatgpt_web2api.cdp_driver import projection_tail_delay
from chatgpt_web2api.provider_contract import BackendHTTPError

@pytest.mark.parametrize('hint,expected',[(20,20.0),(30,30.0),(120,120.0),(1200,1200.0)])
def test_projection_delay_does_not_undercut_retry_after(hint,expected):
    delay,streak,limited=projection_tail_delay('fetch_failed',{'rate_limited':True,'retry_after_seconds':hint},0)
    assert delay==expected and streak==1 and limited

@pytest.mark.asyncio
async def test_header_is_preserved_and_read_gate_suppresses_early_repoll(monkeypatch):
    now=[100.0]
    monkeypatch.setattr('chatgpt_web2api.backend_client.time.monotonic',lambda:now[0])
    driver=SimpleNamespace(ensure_token=AsyncMock(),_access_token='test-only',_breakers=None,
      _js_with_data_strict=AsyncMock(side_effect=[json.dumps({'__status':429,'retry_after':'20'}),json.dumps({'nodes':{},'current_node':None})]))
    client=BackendClient(driver)
    with pytest.raises(BackendHTTPError) as first:
        await client._fetch_recent_conversation_projection('original-conversation')
    assert first.value.status==429 and float(first.value.retry_after)==20.0
    now[0]=119.5
    with pytest.raises(BackendHTTPError) as early:
        await client._fetch_recent_conversation_projection('another-conversation-same-account')
    assert float(early.value.retry_after)>=0.5
    assert driver._js_with_data_strict.await_count==1
    now[0]=120.01
    result=await client._fetch_recent_conversation_projection('original-conversation')
    assert result=={'nodes':{},'current_node':None}
    assert driver._js_with_data_strict.await_count==2

@pytest.mark.asyncio
@pytest.mark.parametrize('method',['_fetch_text_for_turn','_fetch_end_turn_for_turn'])
async def test_read_wrappers_preserve_retry_header_diagnostic(method):
    client=BackendClient(object())
    client._fetch_recent_conversation_projection=AsyncMock(side_effect=BackendHTTPError('projection original-conversation',429,retry_after='20'))
    extra={'had_non_text_content':False} if method.endswith('end_turn_for_turn') else {}
    result=await getattr(client,method)('original-conversation',object(),**extra)
    assert result.status=='fetch_failed'
    assert result.diagnostic['rate_limited'] is True
    assert result.diagnostic['retry_after_seconds']==20.0
    assert projection_tail_delay(result.status,result.diagnostic,0)[0]>=20

@pytest.mark.parametrize('hint',[None,'garbage','-5','nan','inf',True])
def test_invalid_retry_hint_does_not_corrupt_bounded_default(hint):
    from chatgpt_web2api.backend_client import projection_retry_after_seconds
    assert projection_retry_after_seconds(hint,now=1000.0) is None

@pytest.mark.parametrize('header,expected',[('Thu, 01 Jan 1970 00:17:00 GMT',20.0),('Thu, 01 Jan 1970 00:10:00 GMT',0.0),('0',0.0),('20',20.0)])
def test_retry_http_date_and_delta_seconds(header,expected):
    from chatgpt_web2api.backend_client import projection_retry_after_seconds
    assert projection_retry_after_seconds(header,now=1000.0)==expected

@pytest.mark.asyncio
async def test_retry_header_absent_uses_conservative_sixty_second_read_gate(monkeypatch):
    now=[100.0]
    monkeypatch.setattr('chatgpt_web2api.backend_client.time.monotonic',lambda:now[0])
    driver=SimpleNamespace(ensure_token=AsyncMock(),_access_token='fixture',_breakers=None,_js_with_data_strict=AsyncMock(return_value=json.dumps({'__status':429})))
    client=BackendClient(driver)
    with pytest.raises(BackendHTTPError):await client._fetch_recent_conversation_projection('original')
    now[0]=150.0
    with pytest.raises(BackendHTTPError) as pending:await client._fetch_recent_conversation_projection('original')
    assert float(pending.value.retry_after)>=10.0
    assert driver._js_with_data_strict.await_count==1
