from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

T = TypeVar("T")


class _EndOfQueue:
    pass


_END_OF_QUEUE = _EndOfQueue()


class BufferedIterator(threading.Thread, Generic[T]):
    def __init__(
        self,
        iterator_fn: Callable[[], Iterable[T]],
        buffer_size: int,
        timeout: float = 0.1,
    ) -> None:
        super().__init__()
        self.iterator_fn = iterator_fn
        self.timeout = timeout
        self._exception: Optional[Exception] = None
        self._queue: Queue[Union[_EndOfQueue, T]] = Queue(buffer_size)
        self._termination_event = threading.Event()

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
        else:
            self._queue.put(_END_OF_QUEUE)

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self._exception is not None:
            raise self._exception

    def __enter__(self) -> Iterable[T]:
        self.start()
        while True:
            while True:
                if self._exception is not None:
                    raise self._exception
                try:
                    x = self._queue.get(timeout=self.timeout)
                    break
                except Empty:
                    continue
            if x is _END_OF_QUEUE:
                break
            assert not isinstance(x, _EndOfQueue)
            yield x

    def __exit__(self, *_: Any) -> None:
        if self.is_alive():
            self._termination_event.set()
            self.join()
