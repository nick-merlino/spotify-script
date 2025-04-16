# task_queue.py
import asyncio
import logging
from typing import Callable, Any, Coroutine, Tuple

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, max_retries: int = 3) -> None:
        # Each queue item is a tuple of (task_function, current_retry_count)
        self.queue: asyncio.Queue[Tuple[Callable[[], Coroutine[Any, Any, Any]], int]] = asyncio.Queue()
        self.max_retries = max_retries

    def add_task(self, task_func: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        # Initial retry count is 0
        self.queue.put_nowait((task_func, 0))

    async def worker(self) -> None:
        while True:
            task_func, retries = await self.queue.get()
            try:
                logger.info("Running task from queue...")
                await task_func()
            except Exception as e:
                if retries < self.max_retries:
                    logger.warning(f"Task failed with error {e}; retrying ({retries+1}/{self.max_retries})")
                    self.queue.put_nowait((task_func, retries+1))
                else:
                    logger.error(f"Task failed after {retries} retries: {e}")
            finally:
                self.queue.task_done()

    async def run(self, num_workers: int = 2) -> None:
        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        await self.queue.join()
        for w in workers:
            w.cancel()
