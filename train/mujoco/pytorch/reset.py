"""X11 controller for resetting the visible MuJoCo simulator."""

import ctypes
import time


# Simulator reset and target environment.
class MujocoResetController:
    """Find the MuJoCo window and send its Backspace reset shortcut."""

    def __init__(self, window_title: str = "MuJoCo", search_timeout: float = 5.0):
        self.window_title = str(window_title)
        self.search_timeout = float(search_timeout)

    @staticmethod
    def _libraries():
        try:
            x11 = ctypes.CDLL("libX11.so.6")
            xtst = ctypes.CDLL("libXtst.so.6")
        except OSError as exc:
            raise RuntimeError(
                "Automatic MuJoCo reset requires libX11 and libXtst."
            ) from exc

        window = ctypes.c_ulong
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = window
        x11.XQueryTree.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.POINTER(window),
            ctypes.POINTER(window),
            ctypes.POINTER(ctypes.POINTER(window)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        x11.XQueryTree.restype = ctypes.c_int
        x11.XFetchName.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        x11.XFetchName.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XSetInputFocus.argtypes = [
            ctypes.c_void_p,
            window,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        x11.XRaiseWindow.argtypes = [ctypes.c_void_p, window]
        x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XKeysymToKeycode.restype = ctypes.c_uint
        x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        xtst.XTestFakeKeyEvent.restype = ctypes.c_int
        return x11, xtst

    @staticmethod
    def _window_name(x11, display, window: int) -> str:
        name = ctypes.c_char_p()
        if not x11.XFetchName(display, window, ctypes.byref(name)) or not name.value:
            return ""
        try:
            return name.value.decode("utf-8", errors="replace")
        finally:
            x11.XFree(name)

    def _find_window(self, x11, display, root: int) -> int | None:
        pending = [int(root)]
        while pending:
            window = pending.pop()
            if self.window_title in self._window_name(x11, display, window):
                return window

            root_return = ctypes.c_ulong()
            parent_return = ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            count = ctypes.c_uint()
            status = x11.XQueryTree(
                display,
                window,
                ctypes.byref(root_return),
                ctypes.byref(parent_return),
                ctypes.byref(children),
                ctypes.byref(count),
            )
            if not status:
                continue
            try:
                pending.extend(int(children[index]) for index in range(count.value))
            finally:
                if children:
                    x11.XFree(children)
        return None

    def reset(self) -> None:
        x11, xtst = self._libraries()
        display = x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError(
                "Cannot open the X11 display for automatic MuJoCo reset. "
                "Set DISPLAY to the display containing the MuJoCo window."
            )
        try:
            deadline = time.monotonic() + self.search_timeout
            window = None
            while window is None and time.monotonic() < deadline:
                root = x11.XDefaultRootWindow(display)
                window = self._find_window(x11, display, root)
                if window is None:
                    time.sleep(0.1)
            if window is None:
                raise RuntimeError(
                    f"MuJoCo window containing {self.window_title!r} was not found."
                )

            # Backspace is MuJoCo's reset shortcut. Focus the simulator before
            # emitting the synthetic key so another terminal cannot consume it.
            x11.XRaiseWindow(display, window)
            x11.XSetInputFocus(display, window, 2, 0)
            # Deliver FocusIn before the key event. If another app was focused,
            # GLFW can otherwise receive press/release before it processes the
            # focus transition and silently discard the reset shortcut.
            x11.XSync(display, 0)
            time.sleep(0.05)
            keycode = x11.XKeysymToKeycode(display, 0xFF08)
            if not keycode:
                raise RuntimeError("X11 could not resolve the Backspace keycode.")
            # Clear a stale synthetic-down state left by an interrupted sender.
            # XTest otherwise emits release/press in an implementation-dependent
            # order and GLFW may never observe a fresh press transition.
            if not xtst.XTestFakeKeyEvent(display, keycode, 0, 0):
                raise RuntimeError("X11 failed to clear the Backspace key state.")
            x11.XSync(display, 0)
            time.sleep(0.02)
            if not xtst.XTestFakeKeyEvent(display, keycode, 1, 0):
                raise RuntimeError("X11 failed to send the Backspace key press.")
            x11.XSync(display, 0)
            time.sleep(0.02)
            if not xtst.XTestFakeKeyEvent(display, keycode, 0, 0):
                raise RuntimeError("X11 failed to send the Backspace key release.")
            x11.XSync(display, 0)
        finally:
            x11.XCloseDisplay(display)

