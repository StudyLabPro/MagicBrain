"""
Neural Digital Twin endpoints.

Student cognitive modeling via spiking neural networks.
Part of MAGIC Level 2 (MetaBrain) — live cognitive state service.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

router = APIRouter()

# In-memory twin registry (production: Redis-backed)
_twins: Dict[str, "object"] = {}


class CreateTwinRequest(BaseModel):
    """Request to create a neural digital twin."""
    student_id: str
    learning_style: str = Field(
        default="adaptive",
        pattern="^(adaptive|visual|kinesthetic|auditory)$",
    )
    use_stdp: bool = False


class TwinInfo(BaseModel):
    """Twin summary information."""
    student_id: str
    learning_style: str
    genome: str
    topics_learned: int
    overall_mastery: float
    last_updated: str


class LearnRequest(BaseModel):
    """Request to learn a topic."""
    topic_id: str
    topic_name: str = ""
    learning_data: str = Field(..., min_length=5)
    steps: int = Field(default=100, ge=10, le=10000)
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)


class LearnResponse(BaseModel):
    """Learning result."""
    student_id: str
    topic_id: str
    mastery_change: float
    new_mastery: float
    initial_loss: Optional[float]
    final_loss: Optional[float]


class MasteryAssessment(BaseModel):
    """Mastery assessment for a topic."""
    topic_id: str
    mastery: float
    confidence: float
    needs_review: bool
    days_since_practice: int
    neural_activity: float


class CognitiveState(BaseModel):
    """Complete cognitive state snapshot."""
    student_id: str
    overall_mastery: float
    topics_learned: int
    learning_velocity: float
    total_learning_events: int
    neural_metrics: Dict
    last_updated: str


class InteractionEvent(BaseModel):
    """Student interaction event from StudyNinja-API."""
    type: str  # "answer_submitted"|"lesson_completed"|"session_start"|"session_end"
    topic_id: Optional[str] = None
    is_correct: Optional[bool] = None
    response_time_ms: Optional[int] = None
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: Optional[str] = None


class CognitivePrediction(BaseModel):
    """Cognitive prediction from twin."""
    predicted_confusion: float
    predicted_attention: float
    optimal_difficulty: float
    recommended_action: str
    neural_metrics: Dict
    mastery_scores: Dict[str, float]


class EnhancedCognitiveState(CognitiveState):
    """Extended cognitive state with predictions."""
    predicted_performance_next: float
    recommended_break_in_minutes: Optional[float] = None
    topic_readiness: Dict[str, float]
    session_events_count: int


class RecommendationRequest(BaseModel):
    """Request for learning recommendations."""
    available_topics: List[str]


class TopicRecommendation(BaseModel):
    """Single topic recommendation."""
    topic_id: str
    priority: float
    mastery: float
    needs_review: bool
    success_probability: float
    recommendation: str


def _get_twin(student_id: str):
    """Get twin or raise 404."""
    if student_id not in _twins:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Twin for student {student_id} not found",
        )
    return _twins[student_id]


@router.post("/", response_model=TwinInfo, status_code=status.HTTP_201_CREATED)
async def create_twin(request: CreateTwinRequest):
    """Create a Neural Digital Twin for a student."""
    from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin

    if request.student_id in _twins:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Twin for student {request.student_id} already exists",
        )

    twin = NeuralDigitalTwin(
        student_id=request.student_id,
        learning_style=request.learning_style,
        use_stdp=request.use_stdp,
    )
    _twins[request.student_id] = twin

    state = twin.get_cognitive_state()
    return TwinInfo(
        student_id=twin.student_id,
        learning_style=twin.learning_style,
        genome=twin.genome,
        topics_learned=state["topics_learned"],
        overall_mastery=state["overall_mastery"],
        last_updated=state["last_updated"],
    )


@router.get("/{student_id}", response_model=TwinInfo)
async def get_twin(student_id: str):
    """Get twin information for a student."""
    twin = _get_twin(student_id)
    state = twin.get_cognitive_state()
    return TwinInfo(
        student_id=twin.student_id,
        learning_style=twin.learning_style,
        genome=twin.genome,
        topics_learned=state["topics_learned"],
        overall_mastery=state["overall_mastery"],
        last_updated=state["last_updated"],
    )


@router.post("/{student_id}/learn", response_model=LearnResponse)
async def learn_topic(student_id: str, request: LearnRequest):
    """Student learns about a topic via their digital twin."""
    twin = _get_twin(student_id)

    # Register topic if new
    if request.topic_id not in twin.topic_neurons:
        name = request.topic_name or request.topic_id
        twin.register_topic(request.topic_id, name)

    result = twin.learn_topic(
        topic_id=request.topic_id,
        learning_data=request.learning_data,
        steps=request.steps,
        difficulty=request.difficulty,
    )

    return LearnResponse(
        student_id=student_id,
        topic_id=request.topic_id,
        mastery_change=float(result["mastery_change"]),
        new_mastery=float(result["new_mastery"]),
        initial_loss=float(result["initial_loss"]) if result["initial_loss"] is not None else None,
        final_loss=float(result["final_loss"]) if result["final_loss"] is not None else None,
    )


@router.get("/{student_id}/mastery/{topic_id}", response_model=MasteryAssessment)
async def assess_mastery(student_id: str, topic_id: str):
    """Assess current mastery level for a topic."""
    twin = _get_twin(student_id)
    assessment = twin.assess_mastery(topic_id)
    return MasteryAssessment(**assessment)


@router.get("/{student_id}/cognitive-state", response_model=CognitiveState)
async def get_cognitive_state(student_id: str):
    """Get complete cognitive state snapshot."""
    twin = _get_twin(student_id)
    state = twin.get_cognitive_state()
    return CognitiveState(**state)


@router.post(
    "/{student_id}/recommendations",
    response_model=List[TopicRecommendation],
)
async def get_recommendations(student_id: str, request: RecommendationRequest):
    """Get personalized learning recommendations."""
    twin = _get_twin(student_id)

    from magicbrain.integration.knowledgebase_client import KnowledgeBaseClient

    client = KnowledgeBaseClient()
    client.twins[student_id] = twin

    recommendations = await client.get_learning_recommendations(
        student_id=student_id,
        available_topics=request.available_topics,
    )

    return [TopicRecommendation(**rec) for rec in recommendations]


@router.get("/{student_id}/predict/{topic_id}")
async def predict_performance(
    student_id: str,
    topic_id: str,
    difficulty: float = 0.5,
):
    """Predict student performance on a task."""
    twin = _get_twin(student_id)
    prediction = twin.predict_performance(topic_id, difficulty)
    return prediction


@router.post("/{student_id}/events", response_model=CognitivePrediction)
async def process_event(student_id: str, event: InteractionEvent):
    """Process interaction event and return cognitive prediction."""
    twin = _get_twin(student_id)
    result = twin.process_interaction_event(event.model_dump())
    pred = result["cognitive_prediction"]
    return CognitivePrediction(
        predicted_confusion=pred["predicted_confusion"],
        predicted_attention=pred["predicted_attention"],
        optimal_difficulty=pred["optimal_difficulty"],
        recommended_action=pred["recommended_action"],
        neural_metrics=result.get("neural_metrics", {}),
        mastery_scores={
            k: float(v) for k, v in result.get("mastery_scores", {}).items()
        },
    )


@router.post("/{student_id}/batch-events", response_model=CognitivePrediction)
async def process_batch_events(student_id: str, events: List[InteractionEvent]):
    """Process multiple events in sequence."""
    twin = _get_twin(student_id)
    result = None
    for event in events:
        result = twin.process_interaction_event(event.model_dump())
    if result is None:
        raise HTTPException(status_code=400, detail="No events provided")
    pred = result["cognitive_prediction"]
    return CognitivePrediction(
        predicted_confusion=pred["predicted_confusion"],
        predicted_attention=pred["predicted_attention"],
        optimal_difficulty=pred["optimal_difficulty"],
        recommended_action=pred["recommended_action"],
        neural_metrics=result.get("neural_metrics", {}),
        mastery_scores={
            k: float(v) for k, v in result.get("mastery_scores", {}).items()
        },
    )


@router.get("/{student_id}/enhanced-state", response_model=EnhancedCognitiveState)
async def get_enhanced_state(student_id: str):
    """Get enhanced cognitive state with SNN predictions."""
    twin = _get_twin(student_id)
    state = twin.get_enhanced_cognitive_state()
    return EnhancedCognitiveState(**state)


@router.get("/{student_id}/optimal-difficulty/{topic_id}")
async def get_optimal_difficulty(student_id: str, topic_id: str):
    """Get recommended difficulty for a topic."""
    twin = _get_twin(student_id)
    difficulty = twin.get_optimal_difficulty(topic_id)
    return {"difficulty": difficulty, "topic_id": topic_id, "student_id": student_id}


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_twin(student_id: str):
    """Delete a student's digital twin."""
    if student_id not in _twins:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Twin for student {student_id} not found",
        )
    del _twins[student_id]
