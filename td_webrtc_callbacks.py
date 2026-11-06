"""Embedded into the component's WebRTC callbacks DAT by build_clean_toe.py.

Native TD APIs: derivative.ca/UserGuide/WebrtcDAT_Class and Video_Stream_In_TOP.
No Python WebRTC packages are installed into TouchDesigner.
"""
import json
import time


def send(data):
    ws = op('websocket1')
    if ws and parent().fetch('connected', False):
        ws.sendText(json.dumps(data))


def info(text):
    par = getattr(parent().par, 'Transportstate', None)
    if par is not None:
        par.val = text


def reset():
    # Clear identity first, so close callbacks cannot stop a replacement session.
    connection = parent().fetch('rtc_connection', None)
    parent().store('rtc_connection', None)
    parent().store('rtc_session', None)
    parent().store('rtc_video_active', False)
    parent().store('rtc_submitted_at', None)
    parent().store('rtc_decoded', 0)
    rtc = op('webrtc1')
    if rtc and connection:
        rtc.closeConnection(connection)
    video = op('webrtc_video')
    if video:
        video.par.active = False


def fallback(message='WebSocket video (WebRTC fallback)'):
    session = parent().fetch('rtc_session', None)
    if session:
        send({'type': 'rtc_stop', 'session': session})
    reset()
    info(message)


def handle(data):
    if data['type'] == 'rtc_closed':
        if data.get('session') not in (None, parent().fetch('rtc_session', None)):
            return
        reset()
        info(str(data.get('status', 'WebSocket video (WebRTC fallback)')))
        return
    if data['type'] != 'rtc_offer':
        return
    selected = getattr(parent().par, 'Videotransport', None)
    if selected is not None and selected.eval() != 'webrtc':
        send({'type': 'rtc_stop', 'session': data['session']})
        return
    reset()
    parent().store('rtc_session', data['session'])
    try:
        rtc = op('webrtc1')
        if not rtc:
            raise RuntimeError('WebRTC DAT unavailable')
        rtc.par.active = True
        # TD has one shared TURN credential pair. Choose the first TURN entry,
        # matching aiortc's single-relay allocation. API keys never reach TD.
        rtc.par.stun = ''
        rtc.par.turn0server = ''
        rtc.par.username = ''
        rtc.par.password = ''
        turn_set = False
        for server in data.get('ice_servers', []):
            urls = server.get('urls', [])
            urls = [urls] if isinstance(urls, str) else urls
            for url in urls:
                if url.startswith('stun:') and not rtc.par.stun.eval():
                    rtc.par.stun = url
                elif url.startswith(('turn:', 'turns:')) and not turn_set:
                    rtc.par.turn0server = url
                    rtc.par.username = server.get('username', '')
                    rtc.par.password = server.get('credential', '')
                    turn_set = True
        connection = rtc.openConnection()
        parent().store('rtc_connection', connection)
        parent().store('rtc_decoded', 0)
        parent().store('rtc_fps_count', 0)
        parent().store('rtc_fps_time', time.perf_counter())
        rtc.setRemoteDescription(connection, 'offer', data['sdp'])
        rtc.createAnswer(connection)
        info('WebRTC negotiating; WebSocket video remains available')
    except Exception:
        fallback('WebSocket video; native WebRTC setup failed')


def current(connection):
    return connection == parent().fetch('rtc_connection', None)


def onAnswer(webrtcDAT, connectionId, localSdp):
    if current(connectionId):
        webrtcDAT.setLocalDescription(connectionId, 'answer', localSdp)
        send({'type': 'rtc_answer', 'session': parent().fetch('rtc_session', None), 'sdp': localSdp})


def onIceCandidate(webrtcDAT, connectionId, candidate, lineIndex, sdpMid):
    if current(connectionId) and candidate:
        send({'type': 'rtc_candidate', 'session': parent().fetch('rtc_session', None),
              'candidate': candidate, 'sdpMLineIndex': lineIndex, 'sdpMid': sdpMid})


def onTrack(webrtcDAT, connectionId, trackId, type):
    if current(connectionId) and type == 'video':
        video = op('webrtc_video')
        video.par.webrtc = webrtcDAT
        video.par.webrtcconnection = connectionId
        video.par.webrtctrack = trackId
        video.par.active = True


def onConnectionStateChange(webrtcDAT, connectionId, newState):
    if current(connectionId) and newState in ('failed', 'closed'):
        fallback()


def onIceConnectionStateChange(webrtcDAT, connectionId, newState):
    if current(connectionId) and newState == 'failed':
        fallback()


def decoded_count(value):
    # Libwebrtc reports may be arrays or dictionaries keyed by report ID.
    if isinstance(value, list):
        return sum(decoded_count(item) for item in value)
    if isinstance(value, dict):
        if value.get('type') == 'inbound-rtp' and value.get('kind', value.get('mediaType')) == 'video':
            return int(value.get('framesDecoded', 0))
        return sum(decoded_count(item) for item in value.values() if isinstance(item, (dict, list)))
    return 0


def tick(now):
    connection = parent().fetch('rtc_connection', None)
    if not connection or now - parent().fetch('rtc_tick', 0.0) < .1:
        return
    parent().store('rtc_tick', now)
    try:
        rtc = op('webrtc1')
        count = decoded_count(json.loads(rtc.getStats(connectionId=connection)))
        previous = parent().fetch('rtc_decoded', 0)
        if count > previous:
            parent().store('rtc_decoded', count)
            parent().store('rtc_video_active', True)
            parent().store('last_frame_receive', now)
            parent().store('can_send', True)
            parent().store('display_pending', True)
            accumulated = parent().fetch('rtc_fps_count', 0) + count - previous
            elapsed = now - parent().fetch('rtc_fps_time', now)
            parent().store('rtc_fps_count', accumulated)
            if elapsed >= 1:
                fps = accumulated / elapsed
                parent().par.Fps = f'{fps:.1f} FPS'
                if parent().fetch('server_ready', False):
                    parent().par.Status = f'Connected ({fps:.1f} FPS)'
                parent().store('rtc_fps_count', 0)
                parent().store('rtc_fps_time', now)
            info('WebRTC / VP8 (receiving)')
        submitted_at = parent().fetch('rtc_submitted_at', None)
        if submitted_at is not None and count == 0 and now - submitted_at > 15:
            fallback('WebSocket video; WebRTC received no decodable video')
    except Exception:
        fallback('WebSocket video; native WebRTC receiver failed')


def onOffer(webrtcDAT, connectionId, localSdp):
    return


def onNegotiationNeeded(webrtcDAT, connectionId):
    return


def onIceCandidateError(webrtcDAT, connectionId, errorText):
    # A single failed STUN/TURN candidate does not imply that every path failed.
    return


def onRemoveTrack(webrtcDAT, connectionId, trackId, type):
    return


def onSignalingStateChange(webrtcDAT, connectionId, newState):
    return


def onIceGatheringStateChange(webrtcDAT, connectionId, newState):
    return
