"""Run a standard vector environment in a fail-fast spawn process."""

from __future__ import annotations

import logging
import multiprocessing
import signal
import traceback
from multiprocessing.connection import wait


class ProcessEnvironmentError(RuntimeError):
    """A remote environment failed or stopped preserving its state."""


logger = logging.getLogger(__name__)


def _error_payload(operation, error):
    return {
        "operation": operation,
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _send(connection, message):
    try:
        connection.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _environment_process_main(connection, environment_factory, factory_args):
    """Construct and serve one environment entirely inside the child."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    environment = None
    close_request_id = None
    close_error = None
    try:
        try:
            environment = environment_factory(*factory_args)
        except BaseException as error:
            _send(connection, (0, "error", _error_payload("startup", error)))
            return

        if not _send(connection, (0, "ready", None)):
            return

        while True:
            try:
                request_id, operation, payload = connection.recv()
            except (EOFError, OSError):
                return

            if operation == "close":
                close_request_id = request_id
                break

            try:
                if operation == "reset":
                    result = environment.reset()
                elif operation == "step":
                    result = environment.step(payload)
                else:
                    raise ValueError(f"Unsupported environment operation {operation!r}")
            except BaseException as error:
                _send(
                    connection,
                    (request_id, "error", _error_payload(operation, error)),
                )
                return

            if not _send(connection, (request_id, "ok", result)):
                return
    finally:
        if environment is not None:
            try:
                environment.close()
            except BaseException as error:
                close_error = _error_payload("close", error)
        if close_request_id is not None:
            if close_error is None:
                _send(connection, (close_request_id, "ok", None))
            else:
                _send(
                    connection,
                    (
                        close_request_id,
                        "error",
                        close_error,
                    ),
                )
        connection.close()


class EnvironmentProcess:
    """Synchronous reset/step client for one state-preserving child process."""

    def __init__(
        self,
        environment_factory,
        factory_args=(),
        *,
        name="environment-worker",
        startup_timeout=300.0,
        request_timeout=120.0,
        shutdown_timeout=30.0,
    ):
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        self._child_connection = child_connection
        self._process = context.Process(
            target=_environment_process_main,
            args=(child_connection, environment_factory, tuple(factory_args)),
            name=name,
        )
        self.name = name
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self._next_request_id = 1
        self._started = False
        self._ready = False
        self._usable = True
        self._closed = False
        self._close_error = None

    @property
    def pid(self):
        return self._process.pid

    @property
    def is_alive(self):
        return self._process.is_alive() if self._started else False

    @property
    def close_error(self):
        """Return a best-effort shutdown error without masking prior failures."""

        return self._close_error

    def start(self):
        if self._closed:
            raise ProcessEnvironmentError(f"{self.name} is closed")
        if self._started:
            return
        self._process.start()
        self._started = True
        self._child_connection.close()

    def wait_until_ready(self):
        if not self._started:
            raise ProcessEnvironmentError(f"{self.name} has not been started")
        if self._ready:
            return
        request_id, status, payload = self._receive(self.startup_timeout)
        if request_id != 0 or status not in {"ready", "error"}:
            self._break(
                f"invalid startup response {(request_id, status)!r}"
            )
        if status == "error":
            self._usable = False
            error = self._remote_error(payload)
            self._stop_process(self.shutdown_timeout)
            raise error
        self._ready = True

    def reset(self):
        return self._request("reset", None)

    def step(self, actions):
        return self._request("step", actions)

    def _request(self, operation, payload, timeout=None):
        if self._closed or not self._usable:
            raise ProcessEnvironmentError(
                f"{self.name} cannot execute {operation}: worker is unavailable"
            )
        if not self._ready:
            raise ProcessEnvironmentError(
                f"{self.name} cannot execute {operation}: worker is not ready"
            )

        request_id = self._next_request_id
        self._next_request_id += 1
        try:
            self._connection.send((request_id, operation, payload))
        except (BrokenPipeError, EOFError, OSError) as error:
            self._usable = False
            self._stop_process()
            raise ProcessEnvironmentError(
                f"{self.name} could not send {operation}: {error}"
            ) from error

        response_id, status, result = self._receive(
            self.request_timeout if timeout is None else float(timeout)
        )
        if response_id != request_id:
            self._break(
                f"response id {response_id} does not match request {request_id}"
            )
        if status == "error":
            self._usable = False
            error = self._remote_error(result)
            self._stop_process(self.shutdown_timeout)
            raise error
        if status != "ok":
            self._break(f"invalid response status {status!r}")
        return result

    def _receive(self, timeout):
        ready = wait(
            [self._connection, self._process.sentinel], timeout=float(timeout)
        )
        if self._connection in ready:
            try:
                response = self._connection.recv()
            except (EOFError, OSError) as error:
                self._usable = False
                if self._process.sentinel in ready:
                    self._process.join()
                else:
                    self._process.join(timeout=0.05)
                exitcode = self._process.exitcode
                self._stop_process()
                if exitcode is not None:
                    raise ProcessEnvironmentError(
                        f"{self.name} exited unexpectedly with code {exitcode}"
                    ) from error
                raise ProcessEnvironmentError(
                    f"{self.name} closed its connection unexpectedly"
                ) from error
            if not isinstance(response, tuple) or len(response) != 3:
                self._break(f"malformed response {response!r}")
            return response
        if self._process.sentinel in ready:
            self._process.join()
            self._usable = False
            raise ProcessEnvironmentError(
                f"{self.name} exited unexpectedly with code "
                f"{self._process.exitcode}"
            )

        self._usable = False
        self._stop_process()
        raise ProcessEnvironmentError(
            f"{self.name} did not respond within {float(timeout):g} seconds"
        )

    def _remote_error(self, payload):
        if not isinstance(payload, dict):
            return ProcessEnvironmentError(
                f"{self.name} returned malformed error {payload!r}"
            )
        return ProcessEnvironmentError(
            f"{self.name} {payload.get('operation', 'operation')} failed with "
            f"{payload.get('type', 'unknown error')}: "
            f"{payload.get('message', '')}\n{payload.get('traceback', '')}"
        )

    def _break(self, detail):
        self._usable = False
        self._stop_process()
        raise ProcessEnvironmentError(f"{self.name} protocol error: {detail}")

    def _stop_process(self, grace_timeout=0.0):
        if not self._started:
            return
        self._process.join(timeout=float(grace_timeout))
        if not self._process.is_alive():
            return
        self._process.terminate()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5.0)

    def close(self):
        if self._closed:
            return
        graceful = False
        try:
            if self._started and self._ready and self._usable and self.is_alive:
                try:
                    self._request("close", None, self.shutdown_timeout)
                    graceful = True
                except ProcessEnvironmentError as error:
                    self._close_error = error
                    logger.warning("%s", error)
        finally:
            self._usable = False
            self._stop_process(self.shutdown_timeout if graceful else 0.0)
            self._connection.close()
            if not self._started:
                self._child_connection.close()
            self._closed = True
