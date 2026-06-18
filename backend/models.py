"""
SQLAlchemy ORM models: a user account and its single loan account.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    loans = relationship(
        "LoanAccount",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="LoanAccount.opened_at.desc()",
    )


class LoanAccount(Base):
    """A loan tied to a user. A user may have many loans over time, but at
    most one with status="active" (enforced in application code, not here)."""

    __tablename__ = "loan_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    status = Column(String, nullable=False, default="active")  # active | completed
    principal_amount = Column(Float, nullable=False)
    annual_rate = Column(Float, nullable=False)  # ГЛП
    apr = Column(Float, nullable=False)  # ГПР
    term_months = Column(Integer, nullable=False)
    monthly_payment = Column(Float, nullable=False)
    total_payable = Column(Float, nullable=False)
    installments_paid = Column(Integer, nullable=False, default=0)
    outstanding_balance = Column(Float, nullable=False)
    next_payment_date = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, default=dt.datetime.utcnow)

    user = relationship("User", back_populates="loans")
    payments = relationship(
        "Payment",
        back_populates="loan",
        cascade="all, delete-orphan",
        order_by="Payment.paid_at",
    )


class Payment(Base):
    """A record of one payment made against a loan (scheduled installment or early payoff)."""

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    loan_id = Column(Integer, ForeignKey("loan_accounts.id"), nullable=False, index=True)

    amount = Column(Float, nullable=False)
    payment_type = Column(String, nullable=False)  # scheduled | payoff
    installment_number = Column(Integer, nullable=True)
    balance_after = Column(Float, nullable=False)
    paid_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    loan = relationship("LoanAccount", back_populates="payments")
