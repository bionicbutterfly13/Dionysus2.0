"""
T039: Curiosity Trigger Service

Implements curiosity triggers per Spec 029.
Spawns background investigation agents when prediction error > threshold.

Key Features:
- Redis queue for non-blocking triggers
- Prediction error detection (expected vs actual score)
- Background agent spawning
- Investigation status tracking
"""

import logging
from typing import Optional
from datetime import datetime
import json

from models.clause.curiosity_models import CuriosityTrigger

logger = logging.getLogger(__name__)


class CuriosityTriggerService:
    """
    Curiosity trigger service for background investigation.

    Per Spec 029:
    - Trigger on prediction_error > threshold (default 0.7)
    - Use Redis queue for non-blocking operation
    - Spawn background investigation agents
    - Track investigation status (queued → investigating → completed)
    """

    def __init__(self, redis_client=None, threshold: float = 0.7):
        """
        Initialize curiosity trigger service.

        Args:
            redis_client: Redis client for queue management
            threshold: Prediction error threshold (default 0.7)
        """
        self.redis = redis_client
        self.threshold = threshold
        self.queue_name = "curiosity_queue"

        logger.info(f"Curiosity trigger service initialized (threshold={threshold})")

    async def trigger(
        self, concept: str, error_magnitude: float
    ) -> Optional[CuriosityTrigger]:
        """
        Trigger curiosity investigation if error > threshold.

        Args:
            concept: Concept that triggered curiosity
            error_magnitude: Prediction error magnitude [0, 1]

        Returns:
            CuriosityTrigger if triggered, None otherwise
        """
        if error_magnitude < self.threshold:
            logger.debug(
                f"Error magnitude {error_magnitude:.3f} below threshold {self.threshold}"
            )
            return None

        # Create trigger
        trigger = CuriosityTrigger(
            trigger_type="prediction_error",
            concept=concept,
            error_magnitude=error_magnitude,
            timestamp=datetime.now(),
            investigation_status="queued",
        )

        # Add to Redis queue
        if self.redis:
            await self._enqueue(trigger)
            logger.info(f"Curiosity triggered for concept: {concept} (error={error_magnitude:.3f})")
        else:
            logger.warning("Redis not configured - curiosity trigger not queued")

        return trigger

    async def _enqueue(self, trigger: CuriosityTrigger) -> None:
        """Add trigger to Redis queue"""
        if not self.redis:
            return

        trigger_json = trigger.model_dump_json()
        await self.redis.lpush(self.queue_name, trigger_json)

    async def dequeue(self) -> Optional[CuriosityTrigger]:
        """
        Dequeue next trigger for investigation.

        Returns:
            CuriosityTrigger or None if queue empty
        """
        if not self.redis:
            return None

        # Pop from queue (blocking with timeout)
        result = await self.redis.brpop(self.queue_name, timeout=1)

        if result:
            _, trigger_json = result
            trigger = CuriosityTrigger.model_validate_json(trigger_json)
            trigger.investigation_status = "investigating"
            return trigger

        return None

    async def mark_completed(self, concept: str) -> None:
        """
        Mark investigation as completed.

        Args:
            concept: Concept that was investigated
        """
        logger.info(f"Investigation completed for concept: {concept}")
        # In real implementation, would update status in database

    async def get_queue_size(self) -> int:
        """
        Get current queue size.

        Returns:
            Number of pending investigations
        """
        if not self.redis:
            return 0

        return await self.redis.llen(self.queue_name)
