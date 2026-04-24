"""兼容入口：请优先使用 ``marl.utils.live_server``。"""
from marl.utils.live_server import LiveFrameStore, LiveHTTPServer

__all__ = ["LiveFrameStore", "LiveHTTPServer"]
