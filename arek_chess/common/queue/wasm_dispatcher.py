# -*- coding: utf-8 -*-
from typing import Callable, Dict

from js import addEventListener
from pyodide.ffi import create_proxy, to_js
from pyodide.ffi.wrappers import add_event_listener


class WasmDispatcher:
    """
    Dispatcher that receives messages from JS in a web worker and dispatches them into proper queues.
    """

    def __init__(self):
        """"""

        self.callbacks: Dict[str, Callable] = {}

        addEventListener("message", create_proxy(self.on_message))

        # alternative, in case there's a problem with the proxy above
        # add_event_listener(self, "message", self.on_message)

    def on_message(self, event):
        # print("received in py worker: ", event.data.to_py())
        event_data = event.data.to_py()

        type_ = event_data["type"]
        if type_ in self.callbacks:
            self.callbacks[type_](event_data)

    def set_port(self, port):
        add_event_listener(port, "message", self.on_message)
