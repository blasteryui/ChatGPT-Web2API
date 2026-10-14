"""A ready old composer is not a verified fresh project landing."""
import json
from unittest.mock import AsyncMock
import pytest
from chatgpt_web2api.cdp_driver import CDPDriver
PROJECT='g-p-6abc5b56d4488191b9b5b5224e3ef10c'
URL='https://chatgpt.com/g/'+PROJECT+'/project'
def state(**kw):
    return json.dumps({'ready':True,'ready_state':'complete','url':URL,'assistant_count':0,'user_count':0,'composer_empty':True,**kw})
async def noop(_):pass
@pytest.mark.asyncio
@pytest.mark.parametrize('override',[
    {'url':'https://chatgpt.com/c/old-conversation'},
    {'url':'https://chatgpt.com/g/g-p-other/project'},
    {'url':'https://not-chatgpt.com/g/'+PROJECT+'/project'},
    {'assistant_count':3}, {'user_count':1}, {'composer_empty':False},
    {'ready':False}, {'ready_state':'loading'}, {'assistant_count':'0'},
])
async def test_never_admit_stale_nonempty_or_unverified_fresh_landing(monkeypatch,override):
    monkeypatch.setattr('chatgpt_web2api.cdp_driver.asyncio.sleep',noop)
    d=CDPDriver(cdp_port=9222);d._current_conv_id='old-conversation';d._cdp=AsyncMock(return_value={})
    d._js=AsyncMock(return_value=state(**override));d._js_strict=d._js
    with pytest.raises(RuntimeError,match='fresh chat landing'):
        await d.navigate_new_chat(gizmo_id=PROJECT)
    assert d._current_conv_id is None
    d._cdp.assert_awaited_once()
@pytest.mark.asyncio
async def test_old_composer_must_disappear_before_two_verified_fresh_probes(monkeypatch):
    monkeypatch.setattr('chatgpt_web2api.cdp_driver.asyncio.sleep',noop)
    d=CDPDriver(cdp_port=9222);d._current_conv_id='old';d._cdp=AsyncMock(return_value={})
    d._js=AsyncMock(side_effect=[state(url='https://chatgpt.com/c/old',assistant_count=4,user_count=1),state(),state()]);d._js_strict=d._js
    await d.navigate_new_chat(gizmo_id=PROJECT)
    assert d._js.await_count==3
    assert d._current_conv_id is None
@pytest.mark.asyncio
async def test_personal_fresh_chat_requires_empty_root_not_existing_conversation(monkeypatch):
    monkeypatch.setattr('chatgpt_web2api.cdp_driver.asyncio.sleep',noop)
    d=CDPDriver(cdp_port=9222);d._cdp=AsyncMock(return_value={})
    d._js=AsyncMock(return_value=state(url='https://chatgpt.com/?model=auto'));d._js_strict=d._js
    await d.navigate_new_chat()
    assert d._js.await_count==2
@pytest.mark.asyncio
async def test_navigation_error_does_not_continue_into_an_old_composer(monkeypatch):
    monkeypatch.setattr('chatgpt_web2api.cdp_driver.asyncio.sleep',noop)
    d=CDPDriver(cdp_port=9222);d._cdp=AsyncMock(return_value={'errorText':'ERR_ABORTED'});d._current_conv_id='old'
    d._js=AsyncMock(return_value=state());d._js_strict=d._js
    with pytest.raises(RuntimeError,match='fresh chat landing'):
        await d.navigate_new_chat(gizmo_id=PROJECT)
    assert d._js.await_count==0
    assert d._current_conv_id is None
