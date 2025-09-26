"""
VisualizationState Model - T021
Flux Self-Teaching Consciousness Emulator

Represents visualization and UI state for consciousness system interface
with user preferences and real-time display configurations.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class ViewType(str, Enum):
    """Available view types"""
    DASHBOARD = "dashboard"
    CONSCIOUSNESS_MAP = "consciousness_map"
    THOUGHTSEED_TRACE = "thoughtseed_trace"
    CONCEPT_GRAPH = "concept_graph"
    JOURNEY_TIMELINE = "journey_timeline"
    CURIOSITY_MISSIONS = "curiosity_missions"
    EVALUATION_RESULTS = "evaluation_results"
    DOCUMENT_LIBRARY = "document_library"


class ThemeType(str, Enum):
    """Available themes"""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    CONSCIOUSNESS = "consciousness"  # Special consciousness-themed UI


class VisualizationMode(str, Enum):
    """Visualization modes"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    SIMULATION = "simulation"


class PanelConfig(BaseModel):
    """Configuration for individual UI panels"""
    panel_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Panel identifier")
    panel_type: str = Field(..., description="Type of panel")
    position: Dict[str, Union[int, float]] = Field(..., description="Panel position (x, y, width, height)")
    visible: bool = Field(default=True, description="Panel visibility")
    collapsed: bool = Field(default=False, description="Panel collapsed state")
    z_index: int = Field(default=1, description="Panel z-order")

    # Panel-specific settings
    data_source: Optional[str] = Field(None, description="Data source for panel")
    refresh_rate: int = Field(default=1000, description="Refresh rate in milliseconds")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class VisualizationState(BaseModel):
    """
    Visualization state model for consciousness system UI.

    Manages UI state, user preferences, and real-time visualization configurations
    for the Flux consciousness development interface.
    """

    # Core Identity
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique state identifier")
    user_id: str = Field(..., description="Associated user ID")
    session_id: Optional[str] = Field(None, description="UI session identifier")
    journey_id: Optional[str] = Field(None, description="Associated journey ID")

    # State Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last state update")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access timestamp")

    # Active View Configuration
    active_view: ViewType = Field(default=ViewType.DASHBOARD, description="Currently active view")
    previous_view: Optional[ViewType] = Field(None, description="Previous active view")
    view_history: List[Dict[str, Any]] = Field(default_factory=list, description="View navigation history")

    # Theme and Appearance
    theme: ThemeType = Field(default=ThemeType.DARK, description="Active UI theme")
    custom_theme_config: Dict[str, str] = Field(default_factory=dict, description="Custom theme configuration")
    font_size: str = Field(default="medium", description="UI font size")
    color_scheme: Dict[str, str] = Field(default_factory=dict, description="Color scheme settings")

    # Layout Configuration
    layout_mode: str = Field(default="flexible", description="Layout mode (fixed, flexible, custom)")
    panel_configs: List[PanelConfig] = Field(default_factory=list, description="Panel configurations")
    sidebar_collapsed: bool = Field(default=False, description="Sidebar collapsed state")
    fullscreen_mode: bool = Field(default=False, description="Fullscreen mode enabled")

    # Visualization Preferences
    visualization_mode: VisualizationMode = Field(default=VisualizationMode.REAL_TIME, description="Current visualization mode")
    animation_enabled: bool = Field(default=True, description="Enable UI animations")
    smooth_transitions: bool = Field(default=True, description="Enable smooth transitions")
    high_performance_mode: bool = Field(default=False, description="High performance rendering mode")

    # Data Display Settings
    consciousness_metrics_visible: bool = Field(default=True, description="Show consciousness metrics")
    thoughtseed_traces_visible: bool = Field(default=True, description="Show thoughtseed traces")
    concept_graph_visible: bool = Field(default=True, description="Show concept relationships")
    attractor_dynamics_visible: bool = Field(default=True, description="Show attractor dynamics")
    memory_context_visible: bool = Field(default=True, description="Show memory context information")

    # Real-time Updates
    real_time_updates_enabled: bool = Field(default=True, description="Enable real-time data updates")
    update_frequency: int = Field(default=1000, description="Update frequency in milliseconds")
    auto_refresh_enabled: bool = Field(default=True, description="Enable automatic data refresh")

    # Filters and Focus
    active_filters: Dict[str, Any] = Field(default_factory=dict, description="Currently active data filters")
    focus_concept_id: Optional[str] = Field(None, description="Currently focused concept ID")
    focus_trace_id: Optional[str] = Field(None, description="Currently focused thoughtseed trace ID")
    time_range_filter: Dict[str, datetime] = Field(default_factory=dict, description="Active time range filter")

    # Notifications and Alerts
    notifications_enabled: bool = Field(default=True, description="Enable UI notifications")
    alert_preferences: Dict[str, bool] = Field(default_factory=dict, description="Alert type preferences")
    notification_position: str = Field(default="top-right", description="Notification display position")

    # Accessibility Settings
    accessibility_mode: bool = Field(default=False, description="Accessibility mode enabled")
    screen_reader_support: bool = Field(default=False, description="Screen reader support enabled")
    keyboard_navigation: bool = Field(default=True, description="Keyboard navigation enabled")
    reduced_motion: bool = Field(default=False, description="Reduced motion for accessibility")

    # Performance Settings
    max_data_points: int = Field(default=1000, description="Maximum data points to display")
    rendering_quality: str = Field(default="high", description="Rendering quality (low, medium, high)")
    lazy_loading_enabled: bool = Field(default=True, description="Enable lazy loading of components")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    evaluation_feedback_enabled: bool = Field(default=True, description="Evaluation feedback collection enabled")
    privacy_mode: bool = Field(default=False, description="Privacy mode enabled")

    # State Persistence
    auto_save_enabled: bool = Field(default=True, description="Enable automatic state saving")
    save_frequency: int = Field(default=30000, description="Auto-save frequency in milliseconds")
    state_version: int = Field(default=1, description="State version number")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def switch_view(self, new_view: ViewType, context_data: Dict[str, Any] = None) -> None:
        """Switch to a new view"""
        # Record view history
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_view": self.active_view.value,
            "to_view": new_view.value,
            "context": context_data or {}
        }
        self.view_history.append(history_entry)

        # Update views
        self.previous_view = self.active_view
        self.active_view = new_view
        self.updated_at = datetime.utcnow()

    def add_panel(self, panel_type: str, position: Dict[str, Union[int, float]],
                  data_source: str = None, **kwargs) -> str:
        """Add new panel to layout"""
        panel = PanelConfig(
            panel_type=panel_type,
            position=position,
            data_source=data_source,
            **kwargs
        )

        self.panel_configs.append(panel)
        self.updated_at = datetime.utcnow()

        return panel.panel_id

    def update_panel(self, panel_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing panel configuration"""
        for panel in self.panel_configs:
            if panel.panel_id == panel_id:
                for key, value in updates.items():
                    if hasattr(panel, key):
                        setattr(panel, key, value)
                self.updated_at = datetime.utcnow()
                return True
        return False

    def remove_panel(self, panel_id: str) -> bool:
        """Remove panel from layout"""
        original_count = len(self.panel_configs)
        self.panel_configs = [p for p in self.panel_configs if p.panel_id != panel_id]

        if len(self.panel_configs) < original_count:
            self.updated_at = datetime.utcnow()
            return True
        return False

    def set_focus(self, focus_type: str, focus_id: str) -> None:
        """Set UI focus to specific element"""
        if focus_type == "concept":
            self.focus_concept_id = focus_id
        elif focus_type == "trace":
            self.focus_trace_id = focus_id

        self.updated_at = datetime.utcnow()

    def clear_focus(self) -> None:
        """Clear all UI focus"""
        self.focus_concept_id = None
        self.focus_trace_id = None
        self.updated_at = datetime.utcnow()

    def apply_filter(self, filter_name: str, filter_value: Any) -> None:
        """Apply data filter"""
        self.active_filters[filter_name] = filter_value
        self.updated_at = datetime.utcnow()

    def remove_filter(self, filter_name: str) -> None:
        """Remove data filter"""
        if filter_name in self.active_filters:
            del self.active_filters[filter_name]
            self.updated_at = datetime.utcnow()

    def clear_filters(self) -> None:
        """Clear all filters"""
        self.active_filters.clear()
        self.updated_at = datetime.utcnow()

    def set_time_range(self, start_time: datetime, end_time: datetime) -> None:
        """Set time range filter"""
        self.time_range_filter = {
            "start": start_time,
            "end": end_time
        }
        self.updated_at = datetime.utcnow()

    def toggle_real_time_mode(self) -> None:
        """Toggle real-time updates"""
        self.real_time_updates_enabled = not self.real_time_updates_enabled
        self.updated_at = datetime.utcnow()

    def update_theme(self, theme: ThemeType, custom_config: Dict[str, str] = None) -> None:
        """Update UI theme"""
        self.theme = theme
        if custom_config:
            self.custom_theme_config.update(custom_config)
        self.updated_at = datetime.utcnow()

    def get_display_summary(self) -> Dict[str, Any]:
        """Get summary of current display state"""
        return {
            "active_view": self.active_view.value,
            "theme": self.theme.value,
            "panels_count": len(self.panel_configs),
            "filters_active": len(self.active_filters),
            "real_time_enabled": self.real_time_updates_enabled,
            "focused_elements": {
                "concept": self.focus_concept_id,
                "trace": self.focus_trace_id
            },
            "visibility": {
                "consciousness_metrics": self.consciousness_metrics_visible,
                "thoughtseed_traces": self.thoughtseed_traces_visible,
                "concept_graph": self.concept_graph_visible,
                "attractor_dynamics": self.attractor_dynamics_visible
            }
        }

    def save_state_snapshot(self, snapshot_name: str) -> str:
        """Save current state as named snapshot"""
        snapshot_id = str(uuid.uuid4())
        # In real implementation, this would persist the state
        self.state_version += 1
        self.updated_at = datetime.utcnow()
        return snapshot_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

    @classmethod
    def create_mock_state(cls, user_id: str, session_id: str = None,
                         journey_id: str = None) -> "VisualizationState":
        """
        Create mock visualization state for development/testing.
        Constitutional compliance: clearly marked as mock data.
        """
        state = cls(
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            journey_id=journey_id,
            mock_data_enabled=True,
            theme=ThemeType.CONSCIOUSNESS,
            custom_theme_config={
                "primary_color": "#2D5AA0",
                "secondary_color": "#96CEB4",
                "accent_color": "#FECA57",
                "background_color": "#0F0F23"
            },
            color_scheme={
                "consciousness": "#E74C3C",
                "attention": "#3498DB",
                "integration": "#2ECC71",
                "emergence": "#F39C12"
            },
            alert_preferences={
                "consciousness_milestones": True,
                "thoughtseed_completion": True,
                "mission_updates": True,
                "system_errors": True
            }
        )

        # Add mock panels
        state.add_panel(
            "consciousness_metrics",
            {"x": 0, "y": 0, "width": 400, "height": 300},
            data_source="real_time_metrics"
        )

        state.add_panel(
            "concept_graph",
            {"x": 400, "y": 0, "width": 600, "height": 400},
            data_source="concept_relationships"
        )

        # Set mock filters
        state.apply_filter("consciousness_level", "developing")
        state.apply_filter("time_window", "24h")

        return state


# Type aliases for convenience
VisualizationStateDict = Dict[str, Any]
VisualizationStateList = List[VisualizationState]