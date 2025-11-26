"""
Pydantic schemas per validare registry.yaml degli agenti distribuiti.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RegistryGroupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    priority: int = Field(ge=0, le=5)
    context: str
    description: Optional[str] = None
    patterns: List[str]
    obey_robots: bool = True
    max_agents: int = Field(default=1, ge=1, le=10)
    schedule_cron: str
    no_private_areas: Optional[bool] = None


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: List[RegistryGroupModel]
