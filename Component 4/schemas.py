"""Pydantic schemas shared across Component 4 (MD-AP2L)."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class LearningStyle(str, Enum):
    VISUAL = "Visual"
    AUDITORY = "Auditory"
    READING = "Reading/Writing"
    KINESTHETIC = "Kinesthetic"


class Interest(str, Enum):
    SPORTS = "Sports"; MUSIC = "Music"; GAMING = "Gaming"
    MOVIES = "Movies"; TECHNOLOGY = "Technology"; NATURE = "Nature"


class Goal(str, Enum):
    EXAM = "Exam Preparation"; DEEP = "Deep Understanding"; QUICK = "Quick Overview"


class Prior(str, Enum):
    BEGINNER = "Beginner"; INTERMEDIATE = "Intermediate"; ADVANCED = "Advanced"


class Motivation(str, Enum):
    HIGH = "High"; MEDIUM = "Medium"; LOW = "Low"


class Emotion(str, Enum):
    CONFIDENT = "Confident"; ANXIOUS = "Anxious"; BORED = "Bored"
    TIRED = "Tired"; EXCITED = "Excited"


class Strategy(str, Enum):
    SIMPLIFY = "simplify"; ANALOGY = "analogy"; WORKED_EXAMPLE = "worked_example"
    HINT = "hint"; QUIZ = "quiz"; VISUAL = "visual"; BREAK = "break"


class PromptTemplate(str, Enum):
    DIRECT = "Direct Explanation"; STEP = "Step-by-Step"
    ANALOGY = "Analogy-Based"; QUESTION = "Question-Based"
    VISUAL = "Visual Description"; SIMPLIFIED = "Simplified"


class CognitiveState(BaseModel):
    attention: float = Field(..., ge=0, le=1)
    fatigue: float = Field(..., ge=0, le=1)
    confusion: float = Field(..., ge=0, le=1)
    readiness: float = Field(..., ge=0, le=1)
    workload: float = Field(..., ge=0, le=1)


class StudentTraits(BaseModel):
    style: LearningStyle
    interest: Interest
    goal: Goal
    prior: Prior
    motivation: Motivation
    emotion: Emotion


class EvaluationSignal(BaseModel):
    pre_score: float = Field(..., ge=0, le=100)
    post_score: float = Field(..., ge=0, le=100)
    learning_gain: Optional[float] = None
    normalized_gain: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def compute(self) -> "EvaluationSignal":
        self.learning_gain = (self.post_score - self.pre_score) / 100.0
        denom = 100.0 - self.pre_score
        self.normalized_gain = (
            (self.post_score - self.pre_score) / denom if denom > 1e-9 else None
        )
        return self


class OptimizationAction(BaseModel):
    strategy: Strategy
    prompt: PromptTemplate
    algorithm: str
    score: float
    rationale: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)