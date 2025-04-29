import abc
import asyncio
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import redis.asyncio as aioredis
from src.config.connections import get_redis_async_connection
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import MemoryError, ErrorCode, convert_exception, BaseError
from src.utils.serialization import serialize, deserialize, SerializationFormat
from src.memory.utils import AsyncLock
logger = get_logger(__name__)
settings = get_settings()

class BaseTaskQueue(abc.ABC):

    @abc.abstractmethod
    async def produce(self, task_data: Dict[str, Any], task_id: Optional[str]=None) -> str:
        pass

    @abc.abstractmethod
    async def consume(self, consumer_name: str, count: int=1, block_ms: int=2000) -> List[Tuple[str, Dict[str, Any]]]:
        pass

    @abc.abstractmethod
    async def acknowledge(self, message_id: str) -> bool:
        pass

    @abc.abstractmethod
    async def add_to_dlq(self, message_id: str, task_data: Dict[str, Any], error_info: Dict) -> bool:
        pass

    @abc.abstractmethod
    async def get_queue_depth(self) -> int:
        pass

    @abc.abstractmethod
    async def get_lock(self, lock_name: str, expire_time: int=30) -> AsyncLock:
        pass

class RedisStreamTaskQueue(BaseTaskQueue):

    def __init__(self, stream_name: str, consumer_group: str, claim_interval_ms: int=60000):
        self.stream_name: str = stream_name
        self.consumer_group: str = consumer_group
        self.claim_interval_ms: int = claim_interval_ms
        self._redis: Optional[aioredis.Redis] = None
        self._last_pending_check: float = 0
        logger.info(f"RedisStreamTaskQueue initialized for stream '{stream_name}', group '{consumer_group}'")

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            try:
                self._redis = await get_redis_async_connection()
                await self._ensure_stream_and_group()
                logger.info('Redis connection established and stream/group ensured for TaskQueue.')
            except Exception as e:
                error: BaseError = convert_exception(e, ErrorCode.REDIS_CONNECTION_ERROR, 'Failed to establish Redis connection or ensure group for TaskQueue')
                error.log_error(logger)
                self._redis = None
                raise error
        if self._redis is None:
            raise MemoryError(code=ErrorCode.REDIS_CONNECTION_ERROR, message='Redis client initialization failed for TaskQueue.')
        return self._redis

    async def _ensure_stream_and_group(self) -> None:
        if not self._redis:
            raise ConnectionError('Redis client not initialized before ensuring stream/group.')
        try:
            await self._redis.xgroup_create(name=self.stream_name, groupname=self.consumer_group, id='0', mkstream=True)
            logger.debug(f"Ensured consumer group '{self.consumer_group}' exists for stream '{self.stream_name}'.")
        except aioredis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                error: BaseError = convert_exception(e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to ensure Redis group '{self.consumer_group}' for stream '{self.stream_name}'")
                error.log_error(logger)
                raise error
            else:
                logger.debug(f"Consumer group '{self.consumer_group}' already exists for stream '{self.stream_name}'.")
        except Exception as e:
            error: BaseError = convert_exception(e, ErrorCode.REDIS_OPERATION_ERROR, f'Unexpected error ensuring Redis group/stream: {str(e)}')
            error.log_error(logger)
            raise error

    async def produce(self, task_data: Dict[str, Any], task_id: Optional[str]=None) -> str:
        redis: aioredis.Redis = await self._get_redis()
        try:
            serialized_data: bytes = await serialize(task_data, format=SerializationFormat.MSGPACK)
            message_id_bytes: bytes = await redis.xadd(self.stream_name, {'task_data': serialized_data}, id=task_id or '*')
            message_id: str = message_id_bytes.decode('utf-8')
            logger.debug(f"Produced task {message_id} to stream '{self.stream_name}' (Size: {len(serialized_data)} bytes)")
            return message_id
        except Exception as e:
            error: BaseError = convert_exception(e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to produce task to stream '{self.stream_name}'")
            error.log_error(logger)
            raise error

    async def consume(self, consumer_name: str, count: int=1, block_ms: int=2000) -> List[Tuple[str, Dict[str, Any]]]:
        redis: aioredis.Redis = await self._get_redis()
        tasks: List[Tuple[str, Dict[str, Any]]] = []
        try:
            current_time_ms: float = time.time() * 1000
            if current_time_ms - self._last_pending_check > self.claim_interval_ms:
                logger.debug(f"Consumer '{consumer_name}' checking for pending messages...")
                await self._claim_pending_messages(consumer_name)
                self._last_pending_check = current_time_ms
            response: Optional[List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]] = await redis.xreadgroup(groupname=self.consumer_group, consumername=consumer_name, streams={self.stream_name: '>'}, count=count, block=block_ms)
            if not response:
                logger.debug(f"No new messages consumed by '{consumer_name}' within block time {block_ms}ms.")
                return []
            stream_data: List[Tuple[bytes, Dict[bytes, bytes]]] = response[0][1]
            for message_id_bytes, message_data_bytes in stream_data:
                message_id_str: str = message_id_bytes.decode('utf-8')
                if b'task_data' in message_data_bytes:
                    try:
                        task_data: Dict[str, Any] = await deserialize(message_data_bytes[b'task_data'], format=SerializationFormat.MSGPACK)
                        if isinstance(task_data, dict):
                            tasks.append((message_id_str, task_data))
                            logger.debug(f"Consumed task {message_id_str} by consumer '{consumer_name}'")
                        else:
                            logger.warning(f'Deserialized data for task {message_id_str} is not a dict ({type(task_data)}). Acknowledging and skipping.')
                            await self.acknowledge(message_id_str)
                    except Exception as e:
                        logger.error(f'Failed to deserialize task data for message {message_id_str}: {str(e)}. Acknowledging and skipping.', exc_info=True)
                        await self.acknowledge(message_id_str)
                else:
                    logger.warning(f"Message {message_id_str} missing 'task_data' field. Acknowledging and skipping.")
                    await self.acknowledge(message_id_str)
            logger.debug(f"Consumer '{consumer_name}' consumed {len(tasks)} tasks.")
            return tasks
        except Exception as e:
            if isinstance(e, (aioredis.ConnectionError, asyncio.TimeoutError)):
                logger.warning(f"Redis connection/timeout error during task consumption by '{consumer_name}': {str(e)}")
                return []
            else:
                error: BaseError = convert_exception(e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to consume tasks by consumer '{consumer_name}'")
                error.log_error(logger)
                raise error

    async def _claim_pending_messages(self, consumer_name: str) -> None:
        redis: aioredis.Redis = await self._get_redis()
        try:
            pending_info: List = await redis.xpending_range(name=self.stream_name, groupname=self.consumer_group, min='-', max='+', count=50)
            claimable_message_ids: List[bytes] = []
            if pending_info:
                for msg_info in pending_info:
                    if len(msg_info) >= 3 and isinstance(msg_info[2], int):
                        msg_id_bytes: bytes = msg_info[0]
                        idle_time_ms: int = msg_info[2]
                        if idle_time_ms > self.claim_interval_ms:
                            claimable_message_ids.append(msg_id_bytes)
            if not claimable_message_ids:
                logger.debug(f"No claimable pending messages found for consumer '{consumer_name}' (threshold: {self.claim_interval_ms}ms).")
                return
            logger.info(f"Consumer '{consumer_name}' found {len(claimable_message_ids)} potentially stalled messages. Attempting to claim...")
            claimed_messages: Optional[List[Tuple[bytes, Dict[bytes, bytes]]]] = await redis.xclaim(name=self.stream_name, groupname=self.consumer_group, consumername=consumer_name, min_idle_time=self.claim_interval_ms, message_ids=claimable_message_ids)
            if claimed_messages:
                claimed_ids = [mid.decode() for mid, _ in claimed_messages]
                logger.info(f"Consumer '{consumer_name}' successfully claimed {len(claimed_messages)} stalled messages: {claimed_ids}")
            else:
                logger.info(f"Claim attempt finished for consumer '{consumer_name}', but no messages were claimed (possibly claimed by others).")
        except Exception as e:
            logger.error(f"Error checking/claiming pending messages for consumer '{consumer_name}': {str(e)}")

    async def acknowledge(self, message_id: str) -> bool:
        redis: aioredis.Redis = await self._get_redis()
        try:
            result: int = await redis.xack(self.stream_name, self.consumer_group, message_id)
            logger.debug(f"Acknowledged task {message_id} in stream '{self.stream_name}' for group '{self.consumer_group}'")
            return result > 0
        except Exception as e:
            error: BaseError = convert_exception(e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to acknowledge task {message_id} in stream '{self.stream_name}'")
            error.log_error(logger)
            return False

    async def add_to_dlq(self, message_id: str, task_data: Dict[str, Any], error_info: Dict) -> bool:
        dlq_stream_name: str = f'{self.stream_name}_dlq'
        redis: aioredis.Redis = await self._get_redis()
        task_id = task_data.get('id', message_id)
        logger.warning(f"Attempting to move failed task {message_id} (Task ID: {task_id}) to DLQ stream '{dlq_stream_name}'")
        try:
            dlq_data: Dict[str, Any] = {'original_message_id': message_id, 'failed_at_timestamp': time.time(), 'failed_in_group': self.consumer_group, 'error_info': error_info, 'original_task_data': task_data}
            serialized_dlq: bytes = await serialize(dlq_data, format=SerializationFormat.MSGPACK)
            await redis.xadd(dlq_stream_name, {'dlq_entry': serialized_dlq})
            logger.info(f"Moved failed task {message_id} (Task ID: {task_id}) to DLQ stream '{dlq_stream_name}'")
            ack_success = await self.acknowledge(message_id)
            if not ack_success:
                logger.error(f'Failed to acknowledge original message {message_id} after moving to DLQ!')
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to move task {message_id} (Task ID: {task_id}) to DLQ stream '{dlq_stream_name}': {str(e)}", exc_info=True)
            return False

    async def get_queue_depth(self) -> int:
        try:
            redis: aioredis.Redis = await self._get_redis()
            pending_summary: Dict[str, Any] = await redis.xpending(self.stream_name, self.consumer_group)
            depth = pending_summary.get('pending', 0)
            logger.debug(f"Current pending message count (queue depth approximation) for group '{self.consumer_group}': {depth}")
            return depth
        except Exception as e:
            logger.warning(f"Could not get stream pending count (depth approximation) for group '{self.consumer_group}': {str(e)}")
            return -1

    async def get_lock(self, lock_name: str, expire_time: int=30) -> AsyncLock:
        redis: aioredis.Redis = await self._get_redis()
        namespaced_lock_name = f'task_queue_lock:{self.stream_name}:{lock_name}'
        logger.debug(f'Providing distributed lock: {namespaced_lock_name}')
        return AsyncLock(redis, namespaced_lock_name, expire_time)