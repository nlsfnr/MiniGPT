from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

T = TypeVar("T")


class ReraisingThread(threading.Thread):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exception: Optional[Exception] = None

    def run(self) -> None:
        try:
            super().run()
        except Exception as e:
            self._exception = e
            raise

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self._exception is not None:
            raise self._exception

    def __enter__(self) -> ReraisingThread:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.join()


class IteratorAsQueue(threading.Thread, Generic[T]):
    def __init__(
        self,
        iterator_fn: Callable[[], Iterable[T]],
        queue: Optional[Queue[T]] = None,
        max_size: int = 0,
        termination_event: Optional[threading.Event] = None,
        timeout: float = 0.1,
    ) -> None:
        super().__init__()
        self.iterator_fn = iterator_fn
        self.timeout = timeout
        self._exception: Optional[Exception] = None
        self._queue = queue or Queue(maxsize=max_size)
        self._termination_event = termination_event or threading.Event()

    def run(self) -> None:
        # Used to break out of the inner loop.
        class StopThread(Exception):
            pass

        try:
            for x in self.iterator_fn():
                while True:
                    if self._termination_event.is_set():
                        raise StopThread()
                    try:
                        self._queue.put(x, timeout=self.timeout)
                        break
                    except Full:
                        pass
        except StopThread:
            pass
        except Exception as e:
            self._exception = e

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self._exception is not None:
            raise self._exception

    def __enter__(
        self,
    ) -> Queue[T]:
        self.start()
        return self._queue

    def __exit__(self, *_: Any) -> None:
        if self.is_alive():
            self._termination_event.set()
        self.join()


def queue_as_iterator(queue: Queue[T]) -> Iterable[T]:
    while True:
        try:
            yield queue.get()
        except Empty:
            break
