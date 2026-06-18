"""
FastAPI backend for the LendRight RAG Chatbot.

Endpoints
---------
GET  /             → health check
GET  /health       → health check (JSON)
POST /chat         → main RAG chat endpoint
GET  /documents    → list all indexed documents (for debugging)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

# ── Load environment variables ────────────────────────────────────────────────

load_dotenv()

import models
import rag_engine
import security
import loan_service
from database import Base, engine, get_db
from loan_seed import generate_loan_account


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation turns (oldest first).",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve.",
    )


class SourceChunk(BaseModel):
    id: str
    category: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    documents_indexed: int


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=20)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=72)


class LoanAccountOut(BaseModel):
    status: str
    principal_amount: float
    annual_rate: float
    apr: float
    term_months: int
    monthly_payment: float
    total_payable: float
    installments_paid: int
    outstanding_balance: float
    next_payment_date: Optional[str]


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    phone: Optional[str]
    loan: Optional[LoanAccountOut]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class LoanApplyRequest(BaseModel):
    principal_amount: float
    term_months: int


class PaymentOut(BaseModel):
    id: int
    amount: float
    payment_type: str
    installment_number: Optional[int]
    balance_after: float
    paid_at: str


class LoanDetailOut(LoanAccountOut):
    id: int
    opened_at: str
    payments: List[PaymentOut]


class ScheduleEntryOut(BaseModel):
    installment_number: int
    due_date: str
    payment_amount: float
    principal_component: float
    interest_component: float
    remaining_balance: float
    is_paid: bool


class ScheduleOut(BaseModel):
    loan_id: int
    entries: List[ScheduleEntryOut]


class PaymentActionOut(BaseModel):
    loan: LoanDetailOut
    payment: PaymentOut


# ── Serialization helpers ─────────────────────────────────────────────────────

def _serialize_loan_account(loan: models.LoanAccount) -> LoanAccountOut:
    return LoanAccountOut(
        status=loan.status,
        principal_amount=loan.principal_amount,
        annual_rate=loan.annual_rate,
        apr=loan.apr,
        term_months=loan.term_months,
        monthly_payment=loan.monthly_payment,
        total_payable=loan.total_payable,
        installments_paid=loan.installments_paid,
        outstanding_balance=loan.outstanding_balance,
        next_payment_date=(
            loan.next_payment_date.strftime("%d.%m.%Y")
            if loan.next_payment_date
            else None
        ),
    )


def _serialize_payment(payment: models.Payment) -> PaymentOut:
    return PaymentOut(
        id=payment.id,
        amount=payment.amount,
        payment_type=payment.payment_type,
        installment_number=payment.installment_number,
        balance_after=payment.balance_after,
        paid_at=payment.paid_at.strftime("%d.%m.%Y %H:%M"),
    )


def _serialize_loan_detail(loan: models.LoanAccount) -> LoanDetailOut:
    base = _serialize_loan_account(loan)
    return LoanDetailOut(
        **base.model_dump(),
        id=loan.id,
        opened_at=loan.opened_at.strftime("%d.%m.%Y"),
        payments=[_serialize_payment(p) for p in loan.payments],
    )


def _serialize_user(user: models.User) -> UserOut:
    loan = loan_service.get_active_loan(user)
    loan_out = _serialize_loan_account(loan) if loan is not None else None
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        loan=loan_out,
    )


def _build_personal_context(user: models.User) -> str:
    """Format the signed-in customer's own loan data for injection into the RAG prompt."""
    lines = [f"Име на клиента: {user.full_name}"]
    loan = loan_service.get_active_loan(user)
    if loan is None:
        lines.append("Клиентът няма активен кредит в системата в момента.")
        return "\n".join(lines)

    status_bg = "активен" if loan.status == "active" else "приключен"
    lines.extend([
        f"Статус на кредита: {status_bg}",
        f"Усвоена сума: €{loan.principal_amount:.2f}",
        f"Срок на кредита: {loan.term_months} месеца",
        f"Фиксиран ГЛП: {loan.annual_rate}% / ГПР: {loan.apr}%",
        f"Месечна вноска: €{loan.monthly_payment:.2f}",
        f"Платени вноски: {loan.installments_paid} от {loan.term_months}",
        f"Оставаща дължима сума: €{loan.outstanding_balance:.2f}",
        "Дата на следващо плащане: "
        + (
            loan.next_payment_date.strftime("%d.%m.%Y")
            if loan.next_payment_date
            else "няма"
        ),
    ])
    return "\n".join(lines)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: the vector store is already built at import time inside rag_engine
    Base.metadata.create_all(bind=engine)
    yield


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LendRight RAG Chatbot API",
    description=(
        "A retrieval-augmented generation chatbot backed by Gemini 2.5 Flash "
        "that answers questions about LendRight Financial's loan products."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LendRight RAG Chatbot API is running. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health():
    """Return service health and basic metadata."""
    return HealthResponse(
        status="ok",
        model="gemini-2.5-flash",
        documents_indexed=len(rag_engine.DOCUMENTS),
    )


@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new account, seed a synthetic loan, and return an access token."""
    email = payload.email.lower()
    if db.query(models.User).filter(models.User.email == email).first():
        raise HTTPException(
            status_code=409,
            detail="Вече съществува потребител с този имейл.",
        )

    user = models.User(
        email=email,
        hashed_password=security.hash_password(payload.password),
        full_name=payload.full_name,
        phone=payload.phone,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    loan = models.LoanAccount(**generate_loan_account(user.id))
    db.add(loan)
    db.commit()
    db.refresh(user)

    token = security.create_access_token(user.id)
    return TokenResponse(access_token=token, user=_serialize_user(user))


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email + password and return an access token."""
    user = db.query(models.User).filter(models.User.email == payload.email.lower()).first()
    if user is None or not security.verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Невалиден имейл или парола.")

    token = security.create_access_token(user.id)
    return TokenResponse(access_token=token, user=_serialize_user(user))


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
async def me(current_user: models.User = Depends(security.get_current_user)):
    """Return the signed-in user's profile and loan summary."""
    return _serialize_user(current_user)


def _get_owned_loan(loan_id: int, current_user: models.User, db: Session) -> models.LoanAccount:
    """Fetch a loan by id, raising 404 if it doesn't exist or 403 if it
    belongs to a different user."""
    loan = db.query(models.LoanAccount).filter(models.LoanAccount.id == loan_id).first()
    if loan is None:
        raise HTTPException(status_code=404, detail="Кредитът не е намерен.")
    if loan.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Нямате достъп до този кредит.")
    return loan


@app.post("/loans/apply", response_model=LoanDetailOut, status_code=201, tags=["Loans"])
async def apply_for_loan(
    payload: LoanApplyRequest,
    current_user: models.User = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """Apply for a new loan. Allowed only if the caller has no active loan."""
    if payload.principal_amount not in loan_service.PRINCIPAL_CHOICES or payload.term_months not in loan_service.TERM_CHOICES:
        raise HTTPException(
            status_code=400,
            detail="Невалидна сума или срок. Изберете от наличните опции.",
        )

    if loan_service.get_active_loan(current_user) is not None:
        raise HTTPException(
            status_code=409,
            detail="Вече имате активен кредит. Изплатете го напълно, преди да кандидатствате за нов.",
        )

    loan = models.LoanAccount(
        **loan_service.create_loan(current_user.id, payload.principal_amount, payload.term_months)
    )
    db.add(loan)
    db.commit()
    db.refresh(loan)
    return _serialize_loan_detail(loan)


@app.get("/loans", response_model=List[LoanDetailOut], tags=["Loans"])
async def list_loans(
    current_user: models.User = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """List all of the caller's loans (active and completed), newest first."""
    loans = (
        db.query(models.LoanAccount)
        .filter(models.LoanAccount.user_id == current_user.id)
        .order_by(models.LoanAccount.opened_at.desc())
        .all()
    )
    return [_serialize_loan_detail(loan) for loan in loans]


@app.get("/loans/{loan_id}/schedule", response_model=ScheduleOut, tags=["Loans"])
async def get_loan_schedule(
    loan_id: int,
    current_user: models.User = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full repayment schedule for a loan, annotated with which installments are already paid."""
    loan = _get_owned_loan(loan_id, current_user, db)
    schedule = loan_service.build_schedule(
        loan.principal_amount, loan.term_months, loan.opened_at, loan.monthly_payment
    )
    entries = [
        ScheduleEntryOut(
            installment_number=entry["installment_number"],
            due_date=entry["due_date"].strftime("%d.%m.%Y"),
            payment_amount=entry["payment_amount"],
            principal_component=entry["principal_component"],
            interest_component=entry["interest_component"],
            remaining_balance=entry["remaining_balance"],
            is_paid=entry["installment_number"] <= loan.installments_paid,
        )
        for entry in schedule
    ]
    return ScheduleOut(loan_id=loan.id, entries=entries)


@app.post("/loans/{loan_id}/pay", response_model=PaymentActionOut, tags=["Loans"])
async def pay_loan_installment(
    loan_id: int,
    current_user: models.User = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """Pay exactly the next scheduled installment on a loan."""
    loan = _get_owned_loan(loan_id, current_user, db)
    try:
        result = loan_service.apply_payment(loan)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    payment = models.Payment(loan_id=loan.id, **result)
    db.add(payment)
    db.commit()
    db.refresh(loan)
    db.refresh(payment)
    return PaymentActionOut(loan=_serialize_loan_detail(loan), payment=_serialize_payment(payment))


@app.post("/loans/{loan_id}/payoff", response_model=PaymentActionOut, tags=["Loans"])
async def payoff_loan(
    loan_id: int,
    current_user: models.User = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """Pay off the full remaining balance of a loan immediately."""
    loan = _get_owned_loan(loan_id, current_user, db)
    try:
        result = loan_service.apply_payoff(loan)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    payment = models.Payment(loan_id=loan.id, **result)
    db.add(payment)
    db.commit()
    db.refresh(loan)
    db.refresh(payment)
    return PaymentActionOut(loan=_serialize_loan_detail(loan), payment=_serialize_payment(payment))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    current_user: Optional[models.User] = Depends(security.get_current_user_optional),
):
    """
    Send a question and receive a grounded answer from Gemini 2.5 Flash.

    The *history* field lets you maintain multi-turn conversations: pass the
    previous turns (role + content) and the model will use them as context.
    Pass an `Authorization: Bearer <token>` header (from /auth/login or
    /auth/register) to let the assistant answer questions about your own
    loan account; it's optional — anonymous requests still work.
    """
    history = (
        [{"role": m.role, "content": m.content} for m in request.history]
        if request.history
        else None
    )

    user_context = _build_personal_context(current_user) if current_user else None

    try:
        t0 = time.perf_counter()
        result = rag_engine.answer(
            question=request.question,
            chat_history=history,
            k=request.top_k,
            user_context=user_context,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model error: {exc}",
        )

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        latency_ms=latency_ms,
    )


@app.get("/documents", tags=["Utility"])
async def list_documents():
    """Return all indexed document chunks (useful for debugging)."""
    return {
        "total": len(rag_engine.DOCUMENTS),
        "documents": rag_engine.DOCUMENTS,
    }
