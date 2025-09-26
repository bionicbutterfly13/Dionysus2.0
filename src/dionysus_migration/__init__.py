"""
Dionysus Legacy Component Migration with ThoughtSeed Enhancement

A distributed background migration system that systematically migrates
highest quality components from legacy Dionysus consciousness system
to Dionysus 2.0 using ThoughtSeed framework patterns.

Features:
- Zero downtime migration with DAEDALUS coordination
- Complete component rewrite using ThoughtSeed patterns
- Individual component approval and rollback capabilities
- Background processing without blocking active development
- Consciousness functionality preservation and enhancement
"""

__version__ = "1.0.0"
__author__ = "Dionysus 2.0 Team"

# Core migration components
from .models import *
from .services import *
from .api import *
from .cli import *

# Framework integrations
from .integrations.thoughtseed_integration import ThoughtSeedIntegration
from .integrations.daedalus_integration import DaedalusIntegration
from .integrations.chimera_integration import ChimeraIntegration

__all__ = [
    "ThoughtSeedIntegration",
    "DaedalusIntegration",
    "ChimeraIntegration"
]