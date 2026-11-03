"""
backend/database.py - Production Database Layer & ORM Models.

Features:
- SQLAlchemy ORM with SQLite WAL (Write-Ahead Logging) mode for high concurrent throughput.
- Full session & historical analysis tracking.
- Automated table creation and connection pooling.
"""

import json
from datetime import datetime
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from .config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Enable WAL (Write-Ahead Logging) and Foreign Keys on SQLite connections
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PlayerModel(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), default="Grassroots Player", index=True)
    dominant_hand = Column(String(20), default="Right-Hand")
    role = Column(String(50), default="All-Rounder")
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisSessionModel(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String(64), primary_key=True, index=True)
    player_name = Column(String(100), default="Grassroots Player", index=True)
    activity_type = Column(String(50), index=True)  # "Bowling" or "Batting"
    media_type = Column(String(20))                 # "video" or "image"
    accuracy_score = Column(Float, nullable=False)
    release_speed_kmh = Column(Float, default=0.0)
    is_legal = Column(Boolean, default=True)
    elbow_extension_deg = Column(Float, default=0.0)
    front_knee_angle_deg = Column(Float, default=0.0)
    matched_legend = Column(String(100), index=True)
    style_category = Column(String(100))
    pitch_zone = Column(String(100), default="")

    # JSON-encoded coaching pillars
    strengths_json = Column(Text, default="[]")
    mistakes_json = Column(Text, default="[]")
    better_options_json = Column(Text, default="[]")
    drills_json = Column(Text, default="[]")

    video_local_path = Column(String(255), nullable=True)
    video_deleted = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(), index=True)

    def to_dict(self):
        return {
            "session_id": self.id,
            "player_name": self.player_name,
            "activity_type": self.activity_type,
            "media_type": self.media_type,
            "accuracy_score": self.accuracy_score,
            "release_speed_kmh": self.release_speed_kmh,
            "is_legal": self.is_legal,
            "elbow_extension_deg": self.elbow_extension_deg,
            "front_knee_angle_deg": self.front_knee_angle_deg,
            "matched_legend": self.matched_legend,
            "style_category": self.style_category,
            "pitch_zone": self.pitch_zone,
            "video_deleted": self.video_deleted,
            "strengths": json.loads(self.strengths_json) if self.strengths_json else [],
            "mistakes": json.loads(self.mistakes_json) if self.mistakes_json else [],
            "better_options": json.loads(self.better_options_json) if self.better_options_json else [],
            "drills": json.loads(self.drills_json) if self.drills_json else [],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ObservationModel(Base):
    __tablename__ = "observations"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    main_observation = Column(Text, nullable=False)
    what_this_means = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.90)
    confidence_label = Column(String(50), default="High Confidence")
    drill_name = Column(String(150), default="")
    safety_guidance = Column(Text, default="")
    next_session_focus = Column(Text, default="")
    is_dismissed = Column(Boolean, default=False)
    user_correction = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())

    def to_dict(self):
        return {
            "observation_id": self.id,
            "session_id": self.session_id,
            "main_observation": self.main_observation,
            "what_this_means": self.what_this_means,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "drill_name": self.drill_name,
            "safety_guidance": self.safety_guidance,
            "next_session_focus": self.next_session_focus,
            "is_dismissed": self.is_dismissed,
            "user_correction": self.user_correction,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class SessionComparisonModel(Base):
    __tablename__ = "session_comparisons"

    id = Column(String(64), primary_key=True, index=True)
    session_a_id = Column(String(64), index=True)
    session_b_id = Column(String(64), index=True)
    days_between = Column(Integer, default=0)
    scorecard_delta = Column(Float, default=0.0)
    overall_progress_status = Column(String(50))
    coach_verdict = Column(Text)
    comparison_data_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now())

    def to_dict(self):
        return {
            "id": self.id,
            "session_a_id": self.session_a_id,
            "session_b_id": self.session_b_id,
            "days_between": self.days_between,
            "scorecard_delta": self.scorecard_delta,
            "overall_progress_status": self.overall_progress_status,
            "coach_verdict": self.coach_verdict,
            "data": json.loads(self.comparison_data_json) if self.comparison_data_json else {},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class MultiDeliverySessionModel(Base):
    __tablename__ = "multi_delivery_sessions"

    id = Column(String(64), primary_key=True, index=True)
    player_name = Column(String(100), default="Grassroots Player")
    delivery_count = Column(Integer, default=0)
    overall_repeatability_score = Column(Float, default=0.0)
    repeatability_label = Column(String(50))
    primary_inconsistency = Column(Text)
    results_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now())

    def to_dict(self):
        return {
            "id": self.id,
            "player_name": self.player_name,
            "delivery_count": self.delivery_count,
            "overall_repeatability_score": self.overall_repeatability_score,
            "repeatability_label": self.repeatability_label,
            "primary_inconsistency": self.primary_inconsistency,
            "results": json.loads(self.results_json) if self.results_json else {},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class GoalModel(Base):
    __tablename__ = "player_goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100), default="Grassroots Player", index=True)
    metric_name = Column(String(100))  # e.g. "front_knee_angle", "elbow_extension"
    target_value = Column(Float)
    current_value = Column(Float)
    is_achieved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now())


# Auto-create tables
Base.metadata.create_all(bind=engine)


def run_sqlite_migrations():
    """Ensures existing SQLite databases get new columns without data loss."""
    if "sqlite" in settings.DATABASE_URL:
        with engine.connect() as conn:
            try:
                result = conn.exec_driver_sql("PRAGMA table_info(analysis_sessions)")
                existing_cols = {row[1] for row in result}
                if "video_local_path" not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE analysis_sessions ADD COLUMN video_local_path VARCHAR(255)")
                if "video_deleted" not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE analysis_sessions ADD COLUMN video_deleted BOOLEAN DEFAULT 0")
                conn.commit()
            except Exception:
                pass

run_sqlite_migrations()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
