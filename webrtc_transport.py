"""Optional BSD aiortc video output; controls/authentication stay on the bridge socket."""
import asyncio
from fractions import Fraction
import importlib.util
import json
import os
import time
import urllib.request
import uuid


def available():
    return all(importlib.util.find_spec(name) is not None for name in ('aiortc', 'av'))


def validate_ice_servers(value):
    """Only operator-supplied configuration is accepted; clients cannot set server URLs."""
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError('ICE configuration must be a list of at most eight servers')
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('Invalid ICE server')
        urls = item.get('urls')
        urls = [urls] if isinstance(urls, str) else urls
        if not isinstance(urls, list) or not 1 <= len(urls) <= 8:
            raise ValueError('Invalid ICE server URLs')
        for url in urls:
            if (not isinstance(url, str) or len(url) > 512 or any(c.isspace() for c in url)
                    or url.split(':', 1)[0] not in ('stun', 'turn', 'turns')
                    or '@' in url or '\x00' in url):
                raise ValueError('Invalid ICE server URL')
        clean = {'urls': urls}
        for key in ('username', 'credential'):
            if key in item:
                if not isinstance(item[key], str) or len(item[key]) > 4096 or '\x00' in item[key]:
                    raise ValueError('Invalid ICE credential')
                clean[key] = item[key]
        result.append(clean)
    return result


def ice_servers():
    """Resolve fresh short-lived relay credentials without publishing API keys."""
    explicit = os.environ.get('BRIDGE_ICE_SERVERS_JSON')
    if explicit:
        return validate_ice_servers(json.loads(explicit))
    key = os.environ.get('CLOUDFLARE_TURN_KEY_ID')
    token = os.environ.get('CLOUDFLARE_TURN_API_TOKEN')
    if key or token:
        if not key or not token or not all(c.isalnum() or c in '-_' for c in key):
            raise ValueError('Set both Cloudflare TURN key ID and API token')
        request = urllib.request.Request(
            f'https://rtc.live.cloudflare.com/v1/turn/keys/{key}/credentials/generate-ice-servers',
            data=json.dumps({'ttl': 86400}).encode(),
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read(65536))
        servers = data.get('iceServers')
        return validate_ice_servers(servers if isinstance(servers, list) else [servers])
    return [{'urls': ['stun:stun.l.google.com:19302']}]


def load_colab_secrets():
    """Opt-in Colab Secrets; never print or persist credentials in a notebook cell."""
    try:
        from google.colab import userdata
    except ImportError:
        return
    for name in ('BRIDGE_ICE_SERVERS_JSON', 'CLOUDFLARE_TURN_KEY_ID', 'CLOUDFLARE_TURN_API_TOKEN'):
        if not os.environ.get(name):
            try:
                value = userdata.get(name)
            except Exception:
                continue
            if value:
                os.environ[name] = value


def make_track():
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import VideoFrame

    class LatestVideoTrack(VideoStreamTrack):
        """One pending frame: replace stale frames while the encoder/network is busy."""
        def __init__(self):
            super().__init__()
            self.queue = asyncio.Queue(maxsize=1)
            self.origin = time.monotonic()
            self.last_pts = -1
            self.dropped = 0
            self.submitted = 0

        def put(self, image):
            if self.readyState != 'live':
                return
            if self.queue.full():
                self.queue.get_nowait()
                self.dropped += 1
            self.queue.put_nowait(image)
            self.submitted += 1

        def clear(self):
            if not self.queue.empty():
                self.queue.get_nowait()

        async def recv(self):
            if self.readyState != 'live':
                raise MediaStreamError
            image = await self.queue.get()
            if image is None:
                raise MediaStreamError
            frame = VideoFrame.from_ndarray(image, format='bgr24')
            self.last_pts = max(self.last_pts + 1, round((time.monotonic() - self.origin) * 90000))
            frame.pts, frame.time_base = self.last_pts, Fraction(1, 90000)
            return frame

        def stop(self):
            super().stop()
            self.clear()
            self.queue.put_nowait(None)

    return LatestVideoTrack()


class WebRTCOutput:
    def __init__(self, send_json):
        self.send_json = send_json
        self.pc = None
        self.track = None
        self.session = None
        self.task = None
        self.watchdog = None
        self.connected = False
        self.closed = False
        self.pending_candidates = []
        self.candidate_count = 0
        self.reason = None

    def start(self):
        self.task = asyncio.create_task(self._start())

    async def _start(self):
        try:
            from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender
            servers = await asyncio.to_thread(ice_servers)
            if self.closed:
                return
            self.session = uuid.uuid4().hex
            self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**s) for s in servers]))
            self.track = make_track()
            transceiver = self.pc.addTransceiver(self.track, direction='sendonly')
            # VP8 is supported by TD/libwebrtc and needs no NVIDIA video encoder (A100 has none).
            codecs = [c for c in RTCRtpSender.getCapabilities('video').codecs if c.mimeType.lower() == 'video/vp8']
            transceiver.setCodecPreferences(codecs)

            @self.pc.on('connectionstatechange')
            async def state_changed():
                self.connected = self.pc.connectionState == 'connected'
                if self.connected:
                    await self.send_json({'type': 'rtc_state', 'session': self.session, 'state': 'connected'})
                elif self.pc.connectionState == 'failed' and not self.closed:
                    await self.fail('WebRTC connection failed; using WebSocket video')

            offer = await self.pc.createOffer()
            await asyncio.wait_for(self.pc.setLocalDescription(offer), timeout=20)
            if self.closed:
                return
            await self.send_json({'type': 'rtc_offer', 'session': self.session,
                                  'sdp': self.pc.localDescription.sdp, 'ice_servers': servers})
            self.watchdog = asyncio.create_task(self._watch())
        except asyncio.CancelledError:
            raise
        except Exception:
            # ICE URLs/credentials and provider response bodies must never enter logs or public errors.
            await self.fail('WebRTC setup unavailable; using WebSocket video. Check relay configuration and dependencies')

    async def _watch(self):
        await asyncio.sleep(25)
        if not self.connected:
            await self.fail('WebRTC timed out; using WebSocket video. Configure a TURN relay for this network')

    async def receive(self, data):
        if data.get('session') != self.session or self.closed or self.pc is None:
            return
        from aiortc import RTCSessionDescription
        from aiortc.sdp import candidate_from_sdp
        if data['type'] == 'rtc_answer':
            sdp = data.get('sdp')
            if not isinstance(sdp, str) or len(sdp) > 65536 or self.pc.remoteDescription is not None:
                raise ValueError('Invalid WebRTC answer')
            await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type='answer'))
            for candidate in self.pending_candidates:
                await self.pc.addIceCandidate(candidate)
            self.pending_candidates.clear()
        elif data['type'] == 'rtc_candidate':
            self.candidate_count += 1
            candidate = data.get('candidate')
            index, mid = data.get('sdpMLineIndex'), data.get('sdpMid')
            if (self.candidate_count > 64 or not isinstance(candidate, str) or len(candidate) > 2048
                    or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 8
                    or not isinstance(mid, str) or len(mid) > 64):
                raise ValueError('Invalid ICE candidate')
            parsed = candidate_from_sdp(candidate.removeprefix('candidate:'))
            parsed.sdpMLineIndex, parsed.sdpMid = index, mid
            if self.pc.remoteDescription is None:
                self.pending_candidates.append(parsed)
            else:
                await self.pc.addIceCandidate(parsed)

    def clear(self):
        if self.track:
            self.track.clear()

    async def fail(self, reason):
        self.reason = reason
        await self.close()
        await self.send_json({'type': 'rtc_closed', 'status': reason, 'session': self.session})

    async def close(self):
        self.closed, self.connected = True, False
        tasks = [t for t in (self.task, self.watchdog) if t and t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.track:
            self.track.stop()
        if self.pc:
            await self.pc.close()
        self.pending_candidates.clear()
