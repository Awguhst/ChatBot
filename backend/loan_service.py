"""
Amortization math and loan lifecycle mutations, shared by the registration
seeder (loan_seed.py) and the live /loans endpoints in main.py. Framework-free
(no FastAPI imports) so it stays easy to reason about and test in isolation.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import models

ANNUAL_RATE = 40.55  # ГЛП
APR = 49.10  # ГПР

PRINCIPAL_CHOICES = [300, 500, 750, 1000, 1500, 2000, 2500, 3000]
TERM_CHOICES = [3, 6, 9, 12, 18]

_MONTHLY_RATE = ANNUAL_RATE / 100 / 12


def compute_amortization(principal: float, term_months: int) -> Dict[str, float]:
    monthly_payment = principal * _MONTHLY_RATE / (1 - (1 + _MONTHLY_RATE) ** -term_months)
    total_payable = monthly_payment * term_months
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payable": round(total_payable, 2),
    }


def build_schedule(
    principal: float,
    term_months: int,
    opened_at: dt.datetime,
    monthly_payment: float,
) -> List[Dict[str, Any]]:
    """Full amortization table for the loan's original term. The final
    installment's principal component absorbs any rounding drift so the
    balance reaches exactly zero."""
    balance = principal
    schedule = []

    for n in range(1, term_months + 1):
        interest_component = round(balance * _MONTHLY_RATE, 2)

        if n == term_months:
            principal_component = round(balance, 2)
            payment_amount = round(principal_component + interest_component, 2)
        else:
            principal_component = round(monthly_payment - interest_component, 2)
            payment_amount = monthly_payment

        balance = max(round(balance - principal_component, 2), 0.0)

        schedule.append({
            "installment_number": n,
            "due_date": opened_at + dt.timedelta(days=30 * n),
            "payment_amount": payment_amount,
            "principal_component": principal_component,
            "interest_component": interest_component,
            "remaining_balance": balance,
        })

    return schedule


def create_loan(user_id: int, principal_amount: float, term_months: int) -> Dict[str, Any]:
    amort = compute_amortization(principal_amount, term_months)
    return {
        "user_id": user_id,
        "status": "active",
        "principal_amount": round(principal_amount, 2),
        "annual_rate": ANNUAL_RATE,
        "apr": APR,
        "term_months": term_months,
        "monthly_payment": amort["monthly_payment"],
        "total_payable": amort["total_payable"],
        "installments_paid": 0,
        "outstanding_balance": amort["total_payable"],
        "next_payment_date": dt.datetime.utcnow() + dt.timedelta(days=30),
    }


def apply_payment(loan: "models.LoanAccount") -> Dict[str, Any]:
    """Mutate `loan` in place for one scheduled installment payment."""
    if loan.status != "active":
        raise ValueError("Кредитът не е активен.")
    if loan.installments_paid >= loan.term_months:
        raise ValueError("Кредитът няма оставащи вноски.")

    installment_number = loan.installments_paid + 1
    payment_amount = loan.monthly_payment
    if installment_number == loan.term_months:
        payment_amount = loan.outstanding_balance

    loan.installments_paid = installment_number
    loan.outstanding_balance = round(max(loan.outstanding_balance - payment_amount, 0.0), 2)

    if loan.installments_paid >= loan.term_months:
        loan.status = "completed"
        loan.next_payment_date = None
    else:
        loan.next_payment_date = (loan.next_payment_date or dt.datetime.utcnow()) + dt.timedelta(days=30)

    return {
        "amount": round(payment_amount, 2),
        "payment_type": "scheduled",
        "installment_number": installment_number,
        "balance_after": loan.outstanding_balance,
    }


def apply_payoff(loan: "models.LoanAccount") -> Dict[str, Any]:
    """Mutate `loan` in place to pay off the remaining balance immediately."""
    if loan.status != "active":
        raise ValueError("Кредитът не е активен.")

    amount = round(loan.outstanding_balance, 2)
    loan.installments_paid = loan.term_months
    loan.outstanding_balance = 0.0
    loan.status = "completed"
    loan.next_payment_date = None

    return {
        "amount": amount,
        "payment_type": "payoff",
        "installment_number": None,
        "balance_after": 0.0,
    }


def get_active_loan(user: "models.User") -> Optional["models.LoanAccount"]:
    for loan in user.loans:
        if loan.status == "active":
            return loan
    return None
